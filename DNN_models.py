#######################################################################
### PYTORCH IMPLEMENTATION OF CONCRETE DROPOUT ########################
### FROM YARIN GAL'S WEBSITE ##########################################
### https://github.com/yaringal/ConcreteDropout/blob/master/concrete-dropout-pytorch.ipynb
### ONLY FOR DENSE LAYERS #############################################
### NEED TO MAKE VERSION FOR SPATIAL ##################################
#######################################################################

import torch
from torch import nn
from torch import optim

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.metrics import mean_squared_error

class BayesFeedForward(nn.Module):
    def __init__(self, nb_features, p=.1):
        super().__init__()
        # super(Model, self).__init__()
        self.linear1 = nn.Linear(1, nb_features)
        # Applies a linear transformation to the incoming data
        # inputs dimension Nx1, output dimension
        # outputs dimension Nxnb_features with N batch size
        self.linear2 = nn.Linear(nb_features, nb_features)
        self.linear3 = nn.Linear(nb_features, nb_features)

        self.linear4_mu = nn.Linear(nb_features, 1)
        self.linear4_logvar = nn.Linear(nb_features, 1)

        self.drop_layer1 = nn.Dropout(p=p)
        self.drop_layer2 = nn.Dropout(p=p)
        self.drop_layer3 = nn.Dropout(p=p)
        self.drop_layer4 = nn.Dropout(p=p)
        self.drop_layer5 = nn.Dropout(p=p)

        self.relu = nn.ReLU()

    def forward(self, x):
        #x1 = nn.Sequential(self.linear1, self.relu)(x)
        #x2 = nn.Sequential(self.linear2, self.relu)(x1)
        #x3 = nn.Sequential(self.linear3, self.relu)(x2)
        
        #mean = nn.Sequential(self.linear4_mu)(x3) # mean=result of forward pass (y)
        #log_var = nn.Sequential(self.linear4_logvar)(x3) 
        
        x1 = nn.Sequential(self.drop_layer1, self.linear1, self.relu)(x)
        x2 = nn.Sequential(self.drop_layer2, self.linear2, self.relu)(x1)
        x3 = nn.Sequential(self.drop_layer3, self.linear3, self.relu)(x2)
        
        mean = nn.Sequential(self.drop_layer4, self.linear4_mu)(x3) # mean=result of forward pass (y)
        log_var = nn.Sequential(self.drop_layer4, self.linear4_logvar)(x3) #log_var=result of forward pass (var of y)

        return mean, log_var #result of forward pass, a version of that, KL sum over all layers


def heteroscedastic_loss(true, mean, log_var): #heteroscedastic allows for different variance (log_var) for each sample y
    precision = torch.exp(-log_var)
    return torch.mean(torch.sum(precision * (true - mean) ** 2 + log_var, 1), 0)

def homoscedastic_loss(true, mean): #heteroscedastic allows for different variance (log_var) for each sample y
    return torch.mean(torch.sum((true - mean) ** 2, 1), 0)


def logsumexp(a):
    a_max = a.max(axis=0)
    return np.log(np.sum(np.exp(a - a_max), axis=0)) + a_max

def test(Y_true, K_test, means, logvar):
    """
    Estimate predictive log likelihood:
    log p(y|x, D) = log int p(y|x, w) p(w|D) dw
                 ~= log int p(y|x, w) q(w) dw
                 ~= log 1/K sum p(y|x, w_k) with w_k sim q(w)
                  = LogSumExp log p(y|x, w_k) - log K
    :Y_true: a 2D array of size N x dim
    :MC_samples: a 3D array of size samples K x N x 2*D
    """
    k = K_test
    N = Y_true.shape[0]
    mean = means
    logvar = logvar
    test_ll = -0.5 * np.exp(-logvar) * (mean - Y_true.squeeze())**2. - 0.5 * logvar - 0.5 * np.log(2 * np.pi) #Y_true[None]
    test_ll = np.sum(np.sum(test_ll, -1), -1)
    test_ll = logsumexp(test_ll) - np.log(k)
    pppp = test_ll / N  # per point predictive probability
    rmse = np.mean((np.mean(mean, 0) - Y_true.squeeze())**2.)**0.5
    return pppp, rmse


class BayesFF_wrap(TransformerMixin):
    def __init__(self, nb_epoch, nb_features, batch_size, l, loss="Euclidean", heteroscedastic=True,p=.1):
        self.nb_epoch = nb_epoch
        self.nb_features = nb_features
        self.batch_size = batch_size
        self.l = l
        self.loss = loss
        self.heteroscedastic = heteroscedastic
        self.wr = 0
        self.dr = 0
        self.p = p

    def fit(self, X, Y):
        self.X = X
        self.Y = Y

        # compute regularizer
        N = self.X.shape[0]
        self.wr = self.l ** 2. / N  # according to the 1/N before the KL term in ELBO

        # model
        self.model = BayesFeedForward(self.nb_features, self.p)
        self.model = self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters()) 
        self.model.train()

        for i in range(self.nb_epoch):
            for batch in range(int(np.ceil(self.X.shape[0] / self.batch_size))):
                _x = self.X[self.batch_size * batch : self.batch_size * (batch+1)]   # new 25/9/2020
                _y = self.Y[self.batch_size * batch : self.batch_size * (batch+1)]    # new 25/9/2020

                x = torch.FloatTensor(_x).cuda()  # 32-bit floating point
                y = torch.FloatTensor(_y).cuda()

                mean, log_var = self.model(x)  # forward pass                
                # Calculate the sum of squared weights
                sum_of_squared_weights = torch.tensor(0.0).cuda()
                for param in self.model.parameters():
                    sum_of_squared_weights += torch.sum(param.data ** 2)
                regularization = sum_of_squared_weights * self.wr / (1 - self.p)

                if self.heteroscedastic:
                    loss = heteroscedastic_loss(y, mean, log_var) + regularization
                else:
                    loss = homoscedastic_loss(y, mean) + regularization

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            perm = torch.randperm(self.X.shape[0])
            self.X = self.X[perm]
            self.Y = self.Y[perm]
        #self.model.eval()

    def predict(self, X):
        # ONLY FOR GRIDSEARCH !!!
        pred_tuple = self.model(torch.FloatTensor(X).cuda())
        pred_tuple = np.asarray(pred_tuple)
        return (pred_tuple)[0].cpu().data.numpy()


    def predict_one(self, X):
        #print("predict")
        pred_tuple = self.model(torch.FloatTensor(X).cuda())
        means, logvars = pred_tuple
        #print(means[:3])
        return pred_tuple

    def predict_K(self, X, K):
        MC_samples = [self.predict_one(X) for _ in range(K)]
        # MC_samples is list of length K_test (=nr samples x_val)
        # every MC_sample[i] is a tuple of 3 (means, logvars, regularization sum)
        # MC_samples[i, 0] is tensor of length = len(X_val): means
        # MC_samples[i, 1] is tensor of length = len(X_val): logvars
        # MC_samples[i, 2] is tensor of length 1: regularization sum
        # mean, log_var, regularization.sum()
        means = torch.stack([tup[0] for tup in MC_samples]).view(K, X.shape[0]).cpu().data.numpy()
        logvar = torch.stack([tup[1] for tup in MC_samples]).view(K, X.shape[0]).cpu().data.numpy()

        return means, logvar

    def get_results(self, X, Y_true, K):
        means, logvar = self.predict_K(X, K)
        pppp, rmse = test(Y_true, K, means, logvar)
        epistemic_uncertainty = np.var(means, 0).mean(0)
        logvar = np.mean(logvar, 0)
        aleatoric_uncertainty = np.exp(logvar).mean(0)
        ps = np.array(
            [torch.sigmoid(module.p_logit).cpu().data.numpy()[0] for module in self.model.modules() if
             hasattr(module, 'p_logit')])
        return means, logvar, rmse, ps, aleatoric_uncertainty, epistemic_uncertainty


    def score(self, X, Y):
        Y_pred, _, _ = self.predict_one(X)
        return mean_squared_error(Y, Y_pred.cpu().data.numpy())**.5

    def get_params(self, deep=True):
        #return {"l": self.l, "model": self.model}          #changed 24/9/2020 11:02 to allow tuning of l
        return {"l": self.l, "nb_epoch": self.nb_epoch,
                "nb_features": self.nb_features,
                "batch_size": self.batch_size}


    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


class Feedforward(nn.Module):
    def __init__(self, nb_features, p=0):
        super().__init__()

        if type(p) == float or type(p) == int:
            p = [p, p, p, p]

        self.linear1 = nn.Linear(1, nb_features)
        self.linear2 = nn.Linear(nb_features, nb_features)
        self.linear3 = nn.Linear(nb_features, nb_features)
        self.linear4 = nn.Linear(nb_features, 1)

        self.drop_layer1 = nn.Dropout(p=p[0])
        self.drop_layer2 = nn.Dropout(p=p[1])
        self.drop_layer3 = nn.Dropout(p=p[2])
        self.drop_layer4 = nn.Dropout(p=p[3])
        # dropout layers automatically divide by (1-p) during training

        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = nn.Sequential(self.drop_layer1, self.linear1, self.relu)(x)
        x2 = nn.Sequential(self.drop_layer2, self.linear2, self.relu)(x1)
        x3 = nn.Sequential(self.drop_layer3, self.linear3, self.relu)(x2)
        mean = nn.Sequential(self.drop_layer4, self.linear4)(x3)
        return mean


def L2_weight_regularizer(model, wr):
    if type(wr) == float or type(wr) == int:
        wr = [wr, wr, wr, wr]

    weight_regularizer = 0
    layer_nr = 0

    for module in model.children():

        if isinstance(module, nn.Linear):
            sum_of_square = 0
            nr_params = 0
            for param in module.parameters():
                sum_of_square += torch.sum(torch.pow(param, 2))
            weight_regularizer += wr[layer_nr] * sum_of_square
            layer_nr += 1
    return weight_regularizer


class FF_sklearn(TransformerMixin):
    def __init__(self, nb_epoch, nb_features, batch_size, p, wr):
        self.nb_epoch = nb_epoch
        self.nb_features = nb_features
        self.batch_size = batch_size
        self.p = p
        self.wr = wr

        # model
        self.model = Feedforward(nb_features, self.p)
        self.model = self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters())

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        for i in range(self.nb_epoch):
            # print("epoch:{}".format(i))
            old_batch = 0
            for batch in range(int(np.ceil(self.X.shape[0] / self.batch_size))):
                batch = (batch + 1)
                _x = self.X[old_batch: self.batch_size * batch]
                _y = self.Y[old_batch: self.batch_size * batch]

                x = torch.FloatTensor(_x).cuda()  # 32-bit floating point
                y = torch.FloatTensor(_y).cuda()

                mean = self.model(x)  # forward pass

                L2 = L2_weight_regularizer(self.model, self.wr)

                loss = heteroscedastic_loss(y, mean, torch.zeros(y.shape).cuda()) + L2

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.model.eval()

    def predict(self, X):
        y_pred = self.model(torch.FloatTensor(X).cuda())
        y_pred = y_pred.cpu().data.numpy()
        return y_pred

    def score(self, X, Y):
        ("use score")
        self.X = X
        self.Y = Y
        self.Y_pred = self.predict(X)
        return mean_squared_error(self.Y, self.Y_pred)

    def get_params(self, deep=True):
        return {"nb_epoch": self.nb_epoch,
                "nb_features": self.nb_features,
                "batch_size": self.batch_size,
                "p": self.p,
                "wr": self.wr}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

################################################################################
########## OLD #########################  OLD ##################################
################################################################################


def fit_BayesFeedForward(nb_epoch, nb_features, batch_size, l, X, Y, loss="Euclidean"):
    # removed 22.9.2020
    N = X.shape[0]
    assert loss in {"Euclidean", "cross-entropy"}
    wr = l ** 2. / N   #according to the 1/N before the KL term in ELBO
    if loss == "Euclidean":
        dr = 2. / N
    else:
        dr = 1 / N
    #
    # (Eq. 2/3 in paper Concrete dropout) divided by N (see Eq. 1)
    # for Euclidian loss, it should be: wr/dr = l**2/2
    # for cross-entropy, it should be: wr/dr = l**2 (factor 2 omitted)
    model = BayesFeedForward(nb_features, wr, dr)
    model = model.cuda()
    optimizer = optim.Adam(model.parameters())

    for i in range(nb_epoch):
        old_batch = 0
        for batch in range(int(np.ceil(X.shape[0] / batch_size))):
            batch = (batch + 1)
            _x = X[old_batch: batch_size * batch]
            _y = Y[old_batch: batch_size * batch]

            x = torch.FloatTensor(_x).cuda()    #32-bit floating point
            y = torch.FloatTensor(_y).cuda()
            # x = Variable(torch.FloatTensor(_x)).cuda()  # Variable() deprecated
            # y = Variable(torch.FloatTensor(_y)).cuda()

            mean, log_var, regularization = model(x)  #forward pass

            loss = heteroscedastic_loss(y, mean, log_var) + regularization

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model