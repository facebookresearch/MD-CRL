import pytorch_lightning as pl
from models.utils import update
import torch
import hydra
from omegaconf import OmegaConf
from torch.nn import functional as F
from models.utils import penalty_loss_minmax, penalty_loss_stddev, hinge_loss, penalty_domain_classification, mmd_loss
import utils.general as utils
log = utils.get_logger(__name__)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score


class BasePl(pl.LightningModule):
    """
    A LightningModule organizes your PyTorch code into 5 sections:
        - Setup for all computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters(ignore=["datamodule", "additional_logger"], logger=False) # This is CRUCIAL, o.w. checkpoints try to pickle 
        # datamodule which not only takes a lot of space, but raises error because in contains generator
        # objects that cannot be pickled.
        if kwargs.get("hparams_overrides", None) is not None:
            # Overriding the hyper-parameters of a checkpoint at an arbitrary depth using a dict structure
            hparams_overrides = self.hparams.pop("hparams_overrides")
            update(self.hparams, hparams_overrides)

        self.num_domains = self.hparams.get("num_domains", 4)
        self.z_dim_invariant_model = self.hparams.get("z_dim_invariant", 4)
        # self.domain_classification_flag = False

        self.penalty_criterion = self.hparams.get("penalty_criterion", {"minmax": 0., "stddev": 0., "mmd":0., "domain_classification": 0.})
        if self.penalty_criterion and self.penalty_criterion["minmax"]:
            self.loss_transform = self.hparams.get("loss_transform", "mse")
        if self.penalty_criterion and self.penalty_criterion["mmd"]:
            self.MMD = hydra.utils.instantiate(self.hparams.mmd_loss)
        if self.penalty_criterion and self.penalty_criterion["domain_classification"]:
            # self.domain_classification_flag = True
            # self.automatic_optimization = False
            from models.modules.multinomial_logreg import LogisticRegressionModel
            from torch import nn
            self.multinomial_logistic_regression = LogisticRegressionModel(self.z_dim_invariant_model, self.num_domains)
            self.multinomial_logistic_regression = self.multinomial_logistic_regression.to(self.device)
            self.domain_classification_loss = nn.CrossEntropyLoss()

        self.top_k = self.hparams.get("top_k", 5)
        self.stddev_threshold = self.hparams.get("stddev_threshold", 0.1)
        self.stddev_eps = self.hparams.get("stddev_eps", 1e-4)
        self.hinge_loss_weight = self.hparams.get("hinge_loss_weight", 0.0)
        self.adversarial_training_reg_coeff = self.hparams.get("adversarial_training_reg_coeff", 0.0)
        self.r2_fit_intercept = self.hparams.get("r2_fit_intercept", True)

    def loss(self, x, x_hat, z_hat, domains):

        if len(x_hat.size()) <= 2:
            # x, x_hat: [batch_size, x_dim]
            reconstruction_loss = F.mse_loss(x_hat, x, reduction="mean")
        else:
            reconstruction_loss = F.mse_loss(x_hat.permute(0, 2, 3, 1), x.permute(0, 2, 3, 1), reduction="mean")
        self.log(f"reconstruction_loss", reconstruction_loss.item(), prog_bar=True)
        penalty_loss_value = torch.tensor(0., device=self.device)
        hinge_loss_value = hinge_loss(z_hat, domains, self.num_domains, self.z_dim_invariant_model, self.stddev_threshold, self.stddev_eps, self.hinge_loss_weight) if self.hinge_loss_weight > 0. else torch.tensor(0., device=x.device)

        if self.penalty_criterion and self.penalty_criterion["minmax"]:
            penalty_loss_args = [self.top_k, self.loss_transform]
            penalty_loss_value_ = penalty_loss_minmax(z_hat, domains, self.num_domains, self.z_dim_invariant_model, *penalty_loss_args)
            penalty_loss_value += penalty_loss_value_
            self.log(f"penalty_loss_minmax", penalty_loss_value_.item(), prog_bar=True)
        if self.penalty_criterion and self.penalty_criterion["stddev"]:
            penalty_loss_args = []
            penalty_loss_value_ = penalty_loss_stddev(z_hat, domains, self.num_domains, self.z_dim_invariant_model, *penalty_loss_args)
            penalty_loss_value += penalty_loss_value_
        if self.penalty_criterion and self.penalty_criterion["mmd"]:
            penalty_loss_args = []
            penalty_loss_value_ = mmd_loss(self.MMD, z_hat, domains, self.num_domains, self.z_dim_invariant_model, *penalty_loss_args)
            penalty_loss_value += penalty_loss_value_
            self.log(f"penalty_loss_mmd", penalty_loss_value_.item(), prog_bar=True)
        if self.penalty_criterion and self.penalty_criterion["domain_classification"]:
            penalty_loss_args = [self.multinomial_logistic_regression, self.domain_classification_loss]
            penalty_loss_value_, _ = penalty_domain_classification(z_hat, domains, self.num_domains, self.z_dim_invariant_model, *penalty_loss_args)
            penalty_loss_value += penalty_loss_value_
        
        penalty_loss_value = penalty_loss_value * self.penalty_weight
        hinge_loss_value = hinge_loss_value * self.hinge_loss_weight
        loss = reconstruction_loss + penalty_loss_value + hinge_loss_value
        return loss, reconstruction_loss, penalty_loss_value, hinge_loss_value

    def compute_r2(self, x, y):

        reg = LinearRegression(fit_intercept=self.r2_fit_intercept).fit(x.detach().cpu().numpy(), y.detach().cpu().numpy())
        r2 = reg.score(x.detach().cpu().numpy(), y.detach().cpu().numpy())
        # compute the mean squared error of the prediction
        mse_loss = F.mse_loss(torch.tensor(reg.predict(x.detach().cpu().numpy()), device=x.device), y)
        return r2, mse_loss
    
    def compute_r2_mlp(self, x, y, hidden_size=32, n_layers=2):

        hidden_layer_sizes = tuple([hidden_size] * n_layers)
        reg = MLPRegressor(random_state=1, max_iter=200, hidden_layer_sizes=hidden_layer_sizes, early_stopping=True).fit(x.detach().cpu().numpy(), y.detach().cpu().numpy())
        r2 = reg.score(x.detach().cpu().numpy(), y.detach().cpu().numpy())
        # compute the mean squared error of the prediction
        mse_loss = F.mse_loss(torch.tensor(reg.predict(x.detach().cpu().numpy()), device=x.device), y)
        # loss_ = reg.loss_
        # r2 = reg.validation_scores_[-1]
        # loss_ = reg.loss_curve_[-1]
        return r2, mse_loss
    
    def compute_acc_logistic_regression(self, x, label):

        clf = LogisticRegression(random_state=0, max_iter=500).fit(x.detach().cpu().numpy(), label.detach().cpu().numpy())
        pred_label = clf.predict(x.detach().cpu().numpy())
        acc = accuracy_score(label.detach().cpu().numpy(), pred_label)
        return acc

    def compute_acc_mlp(self, x, label, hidden_size=32, n_layers=2):

        hidden_layer_sizes = tuple([hidden_size] * n_layers)
        clf = MLPClassifier(random_state=1, max_iter=500, hidden_layer_sizes=hidden_layer_sizes).fit(x.detach().cpu().numpy(), label.detach().cpu().numpy())
        pred_label = clf.predict(x.detach().cpu().numpy())
        acc = accuracy_score(label.detach().cpu().numpy(), pred_label)
        return acc
    
    def configure_optimizers(self):

        
        # if not self.domain_classification_flag:
        params = []
        for param in self.parameters():
            if param.requires_grad:
                params.append(param)

        optimizer: torch.optim.Optimizer = hydra.utils.instantiate(self.hparams.optimizer
                                                                , params
                                                                )
        
        if self.hparams.get("scheduler_config"):
            # for pytorch scheduler objects, we should use utils.instantiate()
            if self.hparams.scheduler_config.scheduler['_target_'].startswith("torch.optim"):
                scheduler = hydra.utils.instantiate(self.hparams.scheduler_config.scheduler, optimizer)

            # for transformer function calls, we should use utils.call()
            elif self.hparams.scheduler_config.scheduler['_target_'].startswith("transformers"):
                scheduler = hydra.utils.call(self.hparams.scheduler_config.scheduler, optimizer)
            
            else:
                raise ValueError("The scheduler specified by scheduler._target_ is not implemented.")
                
            scheduler_dict = OmegaConf.to_container(self.hparams.scheduler_config.scheduler_dict
                                                    , resolve=True)
            scheduler_dict["scheduler"] = scheduler

            return [optimizer], [scheduler_dict]
        else:
            # no scheduling
            return [optimizer]
    #     else:
    #         ae_params = []
    #         for param in self.model.parameters():
    #             if param.requires_grad:
    #                 ae_params.append(param)
            
    #         classifier_params = []
    #         for param in self.multinomial_logistic_regression.parameters():
    #             if param.requires_grad:
    #                 classifier_params.append(param)

    #         optimizer_ae: torch.optim.Optimizer = hydra.utils.instantiate(self.hparams.optimizer
    #                                                                 , ae_params
    #                                                                 )
    #         optimizer_classifier: torch.optim.Optimizer = hydra.utils.instantiate(self.hparams.optimizer
    #                                                                 , classifier_params
    #                                                                 )
            
    #         if self.hparams.get("scheduler_config"):
    #             if self.hparams.scheduler_config.scheduler['_target_'].startswith("torch.optim"):
    #                 scheduler_ae = hydra.utils.instantiate(self.hparams.scheduler_config.scheduler, optimizer_ae)
    #                 scheduler_classifier = hydra.utils.instantiate(self.hparams.scheduler_config.scheduler, optimizer_classifier)

    #             elif self.hparams.scheduler_config.scheduler['_target_'].startswith("transformers"):
    #                 scheduler_ae = hydra.utils.call(self.hparams.scheduler_config.scheduler, optimizer_ae)
    #                 scheduler_classifier = hydra.utils.call(self.hparams.scheduler_config.scheduler, optimizer_classifier)
                
    #             else:
    #                 raise ValueError("The scheduler specified by scheduler._target_ is not implemented.")
                    
    #             scheduler_dict_ae = OmegaConf.to_container(self.hparams.scheduler_config.scheduler_dict
    #                                                     , resolve=True)
    #             scheduler_dict_ae["scheduler"] = scheduler_ae
    #             scheduler_dict_classifier = OmegaConf.to_container(self.hparams.scheduler_config.scheduler_dict
    #                                                     , resolve=True)
    #             scheduler_dict_classifier["scheduler"] = scheduler_classifier

    #             return [optimizer_ae, optimizer_classifier], [scheduler_dict_ae, scheduler_dict_classifier]
            
    #         else:
    #             # no scheduling
    #             return [optimizer_ae, optimizer_classifier]

    # def manual_adversarial_training_step(self, x, domains, batch_idx):
    #     optimizer_ae, optimizer_classifier = self.optimizers()

    #     z_hat, x_hat = self(x)

    #     # ae_loss = reconstruction_loss + penalty_loss + hinge_loss
    #     # penalty_loss = min-max * 1(if minmax) + stddev * 1(if stddev) + domain_classification * 1(if domain_classification)
    #     ae_loss, reconstruction_loss, penalty_loss_value, hinge_loss_value = self.loss(x, x_hat, z_hat, domains)

    #     classifier_loss, domain_clf_acc = penalty_domain_classification(z_hat, domains, self.num_domains, self.z_dim_invariant_model, self.multinomial_logistic_regression, self.domain_classification_loss)
    #     # penalty_domain_classification already outputs the negative of cross entropy, so that should be adjusted
    #     classifier_loss = -classifier_loss
    #     # code.interact(local=locals())
    #     # training the autoencoder
    #     self.toggle_optimizer(optimizer_ae)
        

    #     # Regularization term
    #     grad_z_hat = torch.autograd.grad(classifier_loss, z_hat, retain_graph=True)
    #     regularization_loss = sum(torch.norm(grad, 2) ** 2 for grad in grad_z_hat)
    #     ae_loss = ae_loss + regularization_loss * self.adversarial_training_reg_coeff

    #     self.manual_backward(ae_loss, retain_graph=True)
    #     optimizer_ae.step()
    #     optimizer_ae.zero_grad()
    #     self.untoggle_optimizer(optimizer_ae)

    #     # training the classifier (adversary)
    #     self.toggle_optimizer(optimizer_classifier)
    #     self.manual_backward(classifier_loss)
    #     optimizer_classifier.step()
    #     optimizer_classifier.zero_grad()
    #     self.untoggle_optimizer(optimizer_classifier)

    #     if batch_idx % 20 == 0:
    #         log.info(f"x.max(): {x.max()}, x_hat.max(): {x_hat.max()}, x.min(): {x.min()}, x_hat.min(): {x_hat.min()}, x.mean(): {x.mean()}, x_hat.mean(): {x_hat.mean()}")
    #         log.info(f"z_hat.max(): {z_hat.max()}, z_hat.min(): {z_hat.min()}, z_hat.mean(): {z_hat.mean()}")
    #     self.log(f"train_reconstruction_loss", reconstruction_loss.item(), prog_bar=True)
    #     self.log(f"train_regularization", regularization_loss.item(), prog_bar=True)
    #     self.log(f"train_domain_clf_acc", domain_clf_acc.item(), prog_bar=True)
    #     self.log(f"train_penalty_loss", penalty_loss_value.item())
    #     self.log(f"train_hinge_loss", hinge_loss_value.item())
    #     self.log(f"train_loss_ae", ae_loss.item(), prog_bar=True)
    #     self.log(f"train_loss_clf", classifier_loss.item(), prog_bar=True)
