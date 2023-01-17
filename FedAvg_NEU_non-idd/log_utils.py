
def save_result(dataset,train_times,test_r2_score ,client_number ,client_rate ,client_epoch , r ,non_idd ,attack):
    import logging
    logging.basicConfig(level=logging.DEBUG, filename='output.log',format='%(asctime)s - %(message)s')
    logger = logging.getLogger(name=__name__)
    log_dataset = "Dataset: %s " % dataset
    log_times = "- Running time: %d:%02d:%02d" % train_times
    log_context = " - Test: R2_score : %4f " % test_r2_score
    log_parameters = " -Client_number : %2d" % client_number + " -Client_rate : %2f" % client_rate  + " -Communicate Round : %2d" % r  + " -Client_epoch : %2d" % client_epoch
    log_idd_mode =  " -distribution mode : %2d" % non_idd
    log_attack_mode = " -distribution mode : %s" % attack
    
    log_context = log_dataset + log_times + log_context + log_parameters + log_idd_mode + log_attack_mode
    logger.info(log_context)