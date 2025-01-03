
import os
import shutil
import random
import numpy as np
import json





def check_dir_exist_or_build(dir_list, force_emptying: bool = False):
    for x in dir_list:
        if not os.path.exists(x):
            os.makedirs(x)
        elif len(os.listdir(x)) > 0:  # not empty
            if force_emptying:
                print("Forcing to erase all contens of {}".format(x))
                shutil.rmtree(x)
                os.makedirs(x)
            else:
                raise FileExistsError
        else:
            continue
        
    
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    
    
def json_dumps_arguments(output_path, args):
    with open(output_path, "w") as f:
        params = vars(args)
        if "device" in params:
            params["device"] = str(params["device"])
        f.write(json.dumps(params, indent=4))