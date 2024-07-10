class Color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def _algorithm_pseudocode(args, logger, prefix=''):
    meta_config = args['meta_config']
    logger.info(prefix + 'Distribute server model weights to selected clients')
    logger.info(prefix + f'{Color.BOLD}Test{Color.END} the server model on the the test dataset with following configurations:')
    for k in meta_config['test_k']:
        logger.info(prefix + f'\tN: {meta_config["test_client_class"]}, K: {k}, Q: {meta_config["test_query_num"]}')
    logger.info(prefix + f'{Color.BOLD}Training{Color.END} each client:')
    logger.info(prefix + f'\tN: {meta_config["train_client_class"]}, K: {meta_config["train_support_num"]}, Q: {meta_config["train_query_num"]}')
    logger.info(prefix + f'\tFor a total of {meta_config["num_train_tasks"]} epochs:')
    logger.info(prefix + f'\t\tTemporarily {Color.BOLD}fine-tune{Color.END} client model classifier with support set (size {meta_config["train_support_num"]}) for {meta_config["test_ft_steps"]} iterations')
    logger.info(prefix + f'\t\tProceed 1 step {Color.BOLD}meta-update{Color.END} based on loss of query set (size {meta_config["train_query_num"]})')
    logger.info(prefix + f'\t\tProceed {Color.BOLD}global to local{Color.END} partial KD by compute MSE loss between logits of client model and server model')
    logger.info(prefix + f'\t\tProceed {Color.BOLD}local to global{Color.END} partial KD by compute InforNCE between hidden embedding of server model and client model\'s main feature')
    logger.info(prefix + f'\t\tProceed main {Color.BOLD}training{Color.END}: 1 epoch GD for server pipeline on support + query set')


def _knowledge_distillation_generator_pseudocode(args, logger, prefix=''):
    KD_config = args['KD_config']
    logger.info(prefix + f'Global model lr: {KD_config["glb_model_lr"]}, Generator model lr: {KD_config["gen_model_lr"]}, lr decay per epoch: {KD_config["lr_decay_per_epoch"]}, weight decay: {KD_config["weight_decay"]}')
    logger.info(prefix + f'For a total of {KD_config["iterations"]} round:')
    logger.info(prefix + f'\tFor {KD_config["g_inner_round"]} round:')
    logger.info(prefix + '\t\tFor each client:')
    logger.info(prefix + f'\t\t\t{Color.BOLD}Generate{Color.END} {KD_config["batch_size"]} samples with generator model')
    logger.info(prefix + f'\t\t\t(MD Loss) Train generator to maximize the discrepancy between server model and client model')
    logger.info(prefix + f'\t\t\t(Classification Loss) Train generator to minimize the classification loss with regard to client model (teacher)')
    logger.info(prefix + f'\t\t\t(Diversity Loss) Train generator to maximize the diversity of generated samples')
    logger.info(prefix + f'\t\t\t{Color.BOLD}Update{Color.END} generator model')
    logger.info(prefix + f'\tFor {KD_config["d_inner_round"]} round:')
    logger.info(prefix + f'\t\t{Color.BOLD}Sample the {KD_config["batch_size"]} fake data{Color.END}, which we just fine-tuned in previous step, from the generator')
    logger.info(prefix + f'\t\t{Color.BOLD}Train server model (Student){Color.END} to minimize the discrepancy between server logits and client logits')


def main_pseudocode(args, logger):
    logger.info('#' * 100)
    logger.info(f'{Color.BOLD}Configuration:{Color.END}')
    logger.info(f'Using model {Color.BOLD}{args["net_config"]["encoder"]}{Color.END}')
    if args['net_config']['pretrained']:
        logger.info(f'Using {Color.BOLD}pretrained model{Color.END} for server and clients')
    else:
        logger.info(f'{Color.BOLD}Random initialize{Color.END} model for server and clients')
    logger.info(f'Total number of clients: {args["n_parties"]}')
    logger.info(f'Number of clients per round: {int(args["n_parties"])}')
    logger.info(f'For a total of {Color.BOLD}{args["meta_steps"]}{Color.END} round:')
    _algorithm_pseudocode(args, logger, '\t')
    logger.info(f'\t{Color.BOLD}Fusing client model{Color.END} by weighted FedAvg Algorithm')
    if args['use_KD_Generator']:
        logger.info(f'\t{Color.BOLD}Local to global knowledge distillation{Color.END} using Generator (FedFTG algorithm).')
        _knowledge_distillation_generator_pseudocode(args, logger, '\t\t')
    logger.info('#' * 100)