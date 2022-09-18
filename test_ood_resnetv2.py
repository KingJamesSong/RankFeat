from utils import log
import resnetv2
import torch
import torchvision as tv
import time
import torchvision.models as models
import numpy as np
import torch.nn as nn
from utils.test_utils import arg_parser, get_measures
import os
import math

from sklearn.linear_model import LogisticRegressionCV
from torch.autograd import Variable
from utils.mahalanobis_lib import get_Mahalanobis_score


def _l2normalize(v, eps=1e-10):
    return v / (torch.norm(v,dim=2,keepdim=True) + eps)


#Power Iteration as SVD substitute for accleration
def power_iteration(A, iter=20):
    u = torch.FloatTensor(1, A.size(1)).normal_(0, 1).view(1,1,A.size(1)).repeat(A.size(0),1,1).to(A)
    v = torch.FloatTensor(A.size(2),1).normal_(0, 1).view(1,A.size(2),1).repeat(A.size(0),1,1).to(A)
    for _ in range(iter):
      v = _l2normalize(u.bmm(A)).transpose(1,2)
      u = _l2normalize(A.bmm(v).transpose(1,2))
    sigma = u.bmm(A).bmm(v)
    sub = sigma * u.transpose(1,2).bmm(v.transpose(1,2))
    return sub

def make_id_ood(args, logger):
    """Returns train and validation datasets."""
    crop = 480

    val_tx = tv.transforms.Compose([
        tv.transforms.Resize((crop, crop)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    in_set = tv.datasets.ImageFolder(args.in_datadir, val_tx)
    out_set = tv.datasets.ImageFolder(args.out_datadir, val_tx)

    logger.info(f"Using an in-distribution set with {len(in_set)} images.")
    logger.info(f"Using an out-of-distribution set with {len(out_set)} images.")

    in_loader = torch.utils.data.DataLoader(
        in_set, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    out_loader = torch.utils.data.DataLoader(
        out_set, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    return in_set, out_set, in_loader, out_loader

#Maximum Logit Score
def iterate_data_msp(data_loader, model):
    confs = []
    m = torch.nn.Softmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            logits = model(x)
            conf, _ = torch.max(m(logits), dim=-1)
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)

#ODIN Score
def iterate_data_odin(data_loader, model, epsilon, temper, logger):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    confs = []
    for b, (x, y) in enumerate(data_loader):
        x = Variable(x.cuda(), requires_grad=True)


        outputs =  model(x)

        maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
        outputs = outputs / temper

        labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(x.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        # Adding small perturbations to images
        tempInputs = torch.add(x.data, -epsilon, gradient)
        outputs = model(Variable(tempInputs))
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

        confs.extend(np.max(nnOutputs, axis=1))
        if b % 100 == 0:
            logger.info('{} batches processed'.format(b))

    return np.array(confs)

#Energy Score
def iterate_data_energy(data_loader, model, temper):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            # compute output, measure accuracy and record loss.
            logits = model(x)
            conf = temper * torch.logsumexp(logits / temper, dim=1)
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)

#Mahalanobis Distance Score
def iterate_data_mahalanobis(data_loader, model, num_classes, sample_mean, precision,
                             num_output, magnitude, regressor, logger):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        if b % 10 == 0:
            logger.info('{} batches processed'.format(b))
        x = x.cuda()

        Mahalanobis_scores = get_Mahalanobis_score(x, model, num_classes, sample_mean, precision, num_output, magnitude)
        scores = -regressor.predict_proba(Mahalanobis_scores)[:, 1]
        confs.extend(scores)
    return np.array(confs)

#GradNorm Score
def iterate_data_gradnorm(data_loader, model, temperature, num_classes):
    confs = []
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        if b % 100 == 0:
            print('{} batches processed'.format(b))
        inputs = Variable(x.cuda(), requires_grad=True)

        model.zero_grad()
        outputs = model(inputs)
        targets = torch.ones((inputs.shape[0], num_classes)).cuda()
        outputs = outputs / temperature
        loss = torch.mean(torch.sum(-targets * logsoftmax(outputs), dim=-1))

        loss.backward()

        layer_grad = model.head.conv.weight.grad.data


        layer_grad_norm = torch.sum(torch.abs(layer_grad)).cpu().numpy()

        confs.append(layer_grad_norm)

    return np.array(confs)

# Our proposed RankFeat Score
def iterate_data_rankfeat(data_loader, model, temperature):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        if b % 100 == 0:
            print('{} batches processed'.format(b))
        inputs = x.cuda()

        #Logit of Block 4 feature
        feat1 = model.module.intermediate_forward(inputs,layer_index=4)
        B, C, H, W = feat1.size()
        feat1 = feat1.view(B, C, H * W)
        u,s,v = torch.linalg.svd(feat1,full_matrices=False)
        feat1 = feat1 - s[:,0:1].unsqueeze(2)*u[:,:,0:1].bmm(v[:,0:1,:])
        #if you want to use PI for acceleration, comment the above 2 lines and uncomment the line below
        #feat1 = feat1 - power_iteration(feat1, iter=20)
        feat1 = feat1.view(B,C,H,W)
        logits1 = model.module.head(model.module.before_head(feat1))

        # Logit of Block 4 feature
        feat2 = model.module.intermediate_forward(inputs, layer_index=3)
        B, C, H, W = feat2.size()
        feat2 = feat2.view(B, C, H * W)
        u, s, v = torch.linalg.svd(feat2,full_matrices=False)
        feat2 = feat2 - s[:, 0:1].unsqueeze(2) * u[:, :, 0:1].bmm(v[:, 0:1, :])
        #if you want to use PI for acceleration, comment the above 2 lines and uncomment the line below
        #feat2 = feat2 - power_iteration(feat2, iter=20)
        feat2 = feat2.view(B, C, H, W)
        feat2 = model.module.body.block4(feat2)
        logits2 = model.module.head(model.module.before_head(feat2))

        #Fusion at the logit space
        logits = (logits1+logits2) / 2
        conf = temperature * torch.logsumexp(logits / temperature, dim=1)
        confs.extend(conf.data.cpu().numpy())

    return np.array(confs)

#ReAct Score
def iterate_data_react(data_loader, model, temperature):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        if b % 100 == 0:
            print('{} batches processed'.format(b))
        inputs = x.cuda()
        feat = model.module.intermediate_forward(inputs,layer_index=4)
        feat = model.module.before_head(feat)
        feat = torch.clip(feat,max=1.25) #threshold computed by 90% percentile of activations
        logits = model.module.head(feat)
        conf = temperature * torch.logsumexp(logits / temperature, dim=1)
        confs.extend(conf.data.cpu().numpy())
    return np.array(confs)

def run_eval(model, in_loader, out_loader, logger, args, num_classes):
    # switch to evaluate mode
    model.eval()

    logger.info("Running test...")
    logger.flush()

    if args.score == 'MSP':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_msp(in_loader, model)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_msp(out_loader, model)
    elif args.score == 'ODIN':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_odin(in_loader, model, args.epsilon_odin, args.temperature_odin, logger)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_odin(out_loader, model, args.epsilon_odin, args.temperature_odin, logger)
    elif args.score == 'Energy':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_energy(in_loader, model, args.temperature_energy)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_energy(out_loader, model, args.temperature_energy)
    elif args.score == 'Mahalanobis':
        sample_mean, precision, lr_weights, lr_bias, magnitude = np.load(
            os.path.join(args.mahalanobis_param_path, 'results.npy'), allow_pickle=True)
        sample_mean = [s.cuda() for s in sample_mean]
        precision = [p.cuda() for p in precision]

        regressor = LogisticRegressionCV(cv=2).fit([[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]],
                                                   [0, 0, 1, 1])

        regressor.coef_ = lr_weights
        regressor.intercept_ = lr_bias

        temp_x = torch.rand(2, 3, 480, 480)
        temp_x = Variable(temp_x).cuda()
        temp_list = model(x=temp_x, layer_index='all')[1]
        num_output = len(temp_list)

        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_mahalanobis(in_loader, model, num_classes, sample_mean, precision,
                                             num_output, magnitude, regressor, logger)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_mahalanobis(out_loader, model, num_classes, sample_mean, precision,
                                              num_output, magnitude, regressor, logger)
    elif args.score == 'GradNorm':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_gradnorm(in_loader, model, args.temperature_gradnorm, num_classes)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_gradnorm(out_loader, model, args.temperature_gradnorm, num_classes)
    elif args.score == 'RankFeat':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_rankfeat(in_loader, model, args.temperature_rankfeat)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_rankfeat(out_loader, model, args.temperature_rankfeat)
    elif args.score == 'React':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_react(in_loader, model, args.temperature_react)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_react(out_loader, model, args.temperature_react)
    else:
        raise ValueError("Unknown score type {}".format(args.score))

    in_examples = in_scores.reshape((-1, 1))
    out_examples = out_scores.reshape((-1, 1))

    auroc, aupr_in, aupr_out, fpr95 = get_measures(in_examples, out_examples)

    logger.info('============Results for {}============'.format(args.score))
    logger.info('AUROC: {}'.format(auroc))
    logger.info('AUPR (In): {}'.format(aupr_in))
    logger.info('AUPR (Out): {}'.format(aupr_out))
    logger.info('FPR95: {}'.format(fpr95))

    logger.flush()


def main(args):
    logger = log.setup_logger(args)

    torch.backends.cudnn.benchmark = True

    if args.score == 'GradNorm':
        args.batch = 1

    in_set, out_set, in_loader, out_loader = make_id_ood(args, logger)

    logger.info(f"Loading model from {args.model_path}")
    #ResNerv2
    model = resnetv2.KNOWN_MODELS[args.model](head_size=len(in_set.classes))
    state_dict = torch.load(args.model_path)
    model.load_state_dict_custom(state_dict['model'])

    if args.score != 'GradNorm':
        model = torch.nn.DataParallel(model)

    model = model.cuda()

    start_time = time.time()
    run_eval(model, in_loader, out_loader, logger, args, num_classes=len(in_set.classes))
    end_time = time.time()

    logger.info("Total running time: {}".format(end_time - start_time))


if __name__ == "__main__":
    parser = arg_parser()

    parser.add_argument("--in_datadir", help="Path to the in-distribution data folder.")
    parser.add_argument("--out_datadir", help="Path to the out-of-distribution data folder.")

    parser.add_argument('--score', choices=['MSP', 'ODIN', 'Energy', 'Mahalanobis', 'GradNorm','RankFeat','React'], default='RankFeat')

    # arguments for ODIN
    parser.add_argument('--temperature_odin', default=1000, type=int,
                        help='temperature scaling for odin')
    parser.add_argument('--epsilon_odin', default=0.0, type=float,
                        help='perturbation magnitude for odin')

    # arguments for Energy
    parser.add_argument('--temperature_energy', default=1, type=int,
                        help='temperature scaling for energy')

    # arguments for Mahalanobis
    parser.add_argument('--mahalanobis_param_path', default='checkpoints/finetune/tune_mahalanobis',
                        help='path to tuned mahalanobis parameters')

    # arguments for GradNorm
    parser.add_argument('--temperature_gradnorm', default=1, type=float,
                        help='temperature scaling for GradNorm')
    # arguments for React
    parser.add_argument('--temperature_react', default=1, type=float,
                        help='temperature scaling for React')
    # arguments for RankFeat
    parser.add_argument('--temperature_rankfeat', default=1, type=float,
                        help='temperature scaling for RankFeat')

    main(parser.parse_args())
