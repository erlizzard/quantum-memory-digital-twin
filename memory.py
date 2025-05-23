from qsi.qsi import QSI
from qsi.helpers import numpy_to_json
from qsi.state import State, StateProp
import time
import numpy as np
import uuid
from vapor_memory import Lambda895Compact, Ladder895, Ladder780, Ladder1529, Lambda895,Lambda795,Lambda780Superradiance\
                        ,Lambda795Compact, Lambda780RydbergSource,Lambda780BEC, Ladder852, Test_Memory
# Initiate QSI object instance
qsi = QSI()
uid = str(uuid.uuid4())

STORAGE_TIME = None #How long we store for  #we then need to convert our storage time into efficiency so we can select the correct parameters
MEMORY_TYPE = None
MEMORY_TRUNCATION = None

T_IN = None
T_OUT = None
KAPPA_E = None
KAPPA_L = None
NB_E = None
NB_L = None


RETRIGGER = True

@qsi.on_message("state_init")
def state_init(msg):
    global MEMORY_TRUNCATION

    """
    Memory has an internal state, which is declared here
    """
    if MEMORY_TRUNCATION is None:
        return {
            "msg_type": "state_init_response",
            "message": f"This component requires the memory truncation to be set."
        }

    state = State(StateProp(
        state_type="internal",
        truncation=MEMORY_TRUNCATION,
        uuid=uid
    ))
    msg = {
        "msg_type": "state_init_response",
        "states": [state.to_message()]
    }
    return msg

@qsi.on_message("param_query")
def param_query(msg):
    """
    This returns the paramters which can be set by the user. In this case it is just storage duration. We could also let them pick the polarisation that we store with? 
    """
    return {
        "msg_type": "param_query_response",
        "params" : {
            "storage_time": "number",
            "memory_type": "string",
            "memory_truncation": "number",
            "t_in": "number",
            "t_out": "number",
            "k_e": "number",
            "k_l": "number",
            "nB_e": "number",
            "nB_l": "number",

            
        }
    }

@qsi.on_message("param_set")
def param_set(msg):
    """
    Memory needs to be told how long the storage duration is. 
    """
    global STORAGE_TIME
    global MEMORY_TYPE
    global MEMORY_TRUNCATION

    global T_IN 
    global T_OUT 
    global KAPPA_E
    global KAPPA_L 
    global NB_E 
    global NB_L 

    params = msg["params"]
    if "storage_time" in params:
        STORAGE_TIME = float(params["storage_time"].get("value"))
    if "memory_type" in params:
        MEMORY_TYPE = params["memory_type"].get("value")
    if "memory_truncation" in params:
        MEMORY_TRUNCATION = int(params["memory_truncation"].get("value"))
    if 't_in' in params and params["t_in"].get("value") is not None:
        T_IN = float(params["t_in"].get("value"))
    if 't_out' in params and params["t_out"].get("value") is not None:
        T_OUT = float(params["t_out"].get("value"))
    if 'k_e' in params and params["k_e"].get("value") is not None: 
        KAPPA_E = float(params["k_e"].get("value"))
    if 'k_l' in params and params["k_l"].get("value") is not None: 
        KAPPA_L = float(params["k_l"].get("value"))  
    if 'nB_e' in params and params["nB_e"].get("value") is not None:  
        NB_E = float(params["nB_e"].get("value"))
    if 'nB_l' in params and params["nB_l"].get("value") is not None:   
        NB_L = float(params["nB_l"].get("value"))    
    return {
        "msg_type": "param_set_response",
    }

@qsi.on_message("channel_query")
def channel_query(msg):
    global STORAGE_TIME
    global MEMORY_TYPE
    global RETRIGGER


    global T_IN 
    global T_OUT 
    global KAPPA_E
    global KAPPA_L 
    global NB_E 
    global NB_L 

    
    state = State.from_message(msg)
    uuid = msg["ports"]["input"]
    op_type = msg["ports"]["op_type"]


    input_props = state.get_props(uuid)
    internal_props = state.get_props(uid)
    #Here we need to put in our condition

    # If storage_time is not given
    if STORAGE_TIME is None:
        return {
            "msg_type": "channel_query_response_multiple",
            "message": f"This component requires the storage time [s] to be set."
        }


    if MEMORY_TYPE == 'Test':
        #This is so we can directly access the eta_values, the other memory devices abstract some more!

        if T_IN is None:
            return {
                "msg_type": "channel_query_response_multiple",
                "message": f"This component requires the read-in transmission [0-1] to be set."
            }
        if T_OUT is None:
            return {
                "msg_type": "channel_query_response_multiple",
                "message": f"This component requires the read-out transmission [0-1] to be set."
            }
        if KAPPA_E is None:
            return {
                "msg_type": "channel_query_response_multiple",
                "message": f"This component requires the early time-bin transmission [0-1] to be set."
            }
        if KAPPA_L is None:
            return {
                "msg_type": "channel_query_response_multiple",
                "message": f"This component requires the late time-bin transmission [0-1] to be set."
            }
        if NB_E is None:
            return {
                "msg_type": "channel_query_response_multiple",
                "message": f"This component requires the early noise photon number to be set."
            }
        if NB_L is None:
            return {
                "msg_type": "channel_query_response_multiple",
                "message": f"This component requires the  late noise photon number to be set."
            }

        mem = Test_Memory(input_props.truncation, internal_props.truncation)
        mem.t_in = T_IN
        mem.t_out = T_OUT
        mem.eta_e = KAPPA_E
        mem.eta_l = KAPPA_L
        mem.nB_e = NB_E
        mem.nB_l = NB_L

    elif MEMORY_TYPE == 'Cs':
        mem = Lambda895(input_props.truncation, internal_props.truncation,STORAGE_TIME)
    elif MEMORY_TYPE == 'Lambda895':
        mem = Lambda895(input_props.truncation, internal_props.truncation,STORAGE_TIME)
    elif MEMORY_TYPE == 'Lambda895Compact':
        mem = Lambda895Compact(input_props.truncation, internal_props.truncation,STORAGE_TIME)
    elif MEMORY_TYPE == 'Ladder895':
        mem = Ladder895(input_props.truncation, internal_props.truncation,STORAGE_TIME)
    elif MEMORY_TYPE == 'Ladder780':
        mem = Ladder780(input_props.truncation, internal_props.truncation,STORAGE_TIME)
    elif MEMORY_TYPE == 'Ladder1529':
        mem = Ladder1529(input_props.truncation, internal_props.truncation,STORAGE_TIME)
    elif MEMORY_TYPE == 'Lambda795':
        mem = Lambda795(input_props.truncation, internal_props.truncation,STORAGE_TIME)
    elif MEMORY_TYPE == 'Lambda780Superradiance':
        mem = Lambda780Superradiance(input_props.truncation, internal_props.truncation,STORAGE_TIME)
    elif MEMORY_TYPE == 'Lambda795Compact':
        mem = Lambda795Compact(input_props.truncation, internal_props.truncation,STORAGE_TIME)
    elif MEMORY_TYPE == 'Lambda780RydbergSource':
        mem = Lambda780RydbergSource(input_props.truncation, internal_props.truncation,STORAGE_TIME)
    elif MEMORY_TYPE == 'Lambda780BEC':
        mem = Lambda780BEC(input_props.truncation, internal_props.truncation,STORAGE_TIME)
    elif MEMORY_TYPE == 'Ladder852':
        mem = Ladder852(input_props.truncation, internal_props.truncation,STORAGE_TIME)
    else:
        return {
            "msg_type": "channel_query_response_multiple",
            "message": f"Please set the memory type to a valid response."
        }

    # Do not compute channel, if mode with wrong properties is given
    if input_props.wavelength != mem.wavelength:
        return {
            "msg_type": "channel_query_response_multiple",
            "message": f"This component only interacts with 1550 nm modes, received {input_props.wavelength}, {type(input_props.wavelength)}"
        }

     #if the bandwidth of the input field is larger than the memory, then we can't store it. 
    if input_props.bandwidth > mem.bandwidth:
        return  {
            "msg_type": "channel_query_response_multiple",
            "message": f"This component only can only store light with a bandwidth below {mem.bandwidth}, received {input_props.bandwidth}, {type(input_props.bandwidth)}"
        }

    if input_props.polarization not in mem.polarization:
        return  {
            "msg_type": "channel_query_response_multiple",
            "message": f"This component only can only store light with polarization {mem.polarization}, received {input_props.polarization}, {type(input_props.bandwidth)}"
        }
    
    if op_type == 'storage':
        kraus_operators = mem.storage_experiment_combined()    
        kraus_indices = [[input_props.uuid,internal_props.uuid],[input_props.uuid]]
        RETRIGGER = False

        return {
            "msg_type": "channel_query_response_multiple",
            "kraus_operators":  [[numpy_to_json(y) for y in x] for x in kraus_operators],
            "kraus_state_indices": kraus_indices,
            "error": 0,
            "retrigger": RETRIGGER,
            "retrigger_time" : mem.retrigger, #in seconds
            "operation_time" : STORAGE_TIME  
        }
    
    elif op_type == 'debug':
        kraus_operators=  mem.storage_noise()

        kraus_indices = [input_props.uuid]
        return {
            "msg_type": "channel_query_response",
            "kraus_operators": [numpy_to_json(x) for x in kraus_operators],
            "kraus_state_indices": kraus_indices,
            "error": 0,
        }
    
    elif op_type == 'retrieval':
        kraus_operators = mem.retreival_experiment_combined()
        kraus_indices = [[internal_props.uuid, input_props.uuid],[ input_props.uuid]]
        RETRIGGER = True
        return {
                "msg_type": "channel_query_response_multiple",
                "kraus_operators": [[numpy_to_json(y) for y in x] for x in kraus_operators],
                "kraus_state_indices": kraus_indices,
                "error": 0,
                "retrigger": RETRIGGER,
                "retrigger_time" : mem.retrigger, #in seconds
                "operation_time" : STORAGE_TIME  
            }
    
    else:
        return {
                    "msg_type": "channel_query_response_multiple",
                    "message": f"Please set operation type: 'storage' or 'retrieval'."
                }
    
@qsi.on_message("terminate")
def terminate(msg):
    qsi.terminate()

qsi.run()
time.sleep(1)

class WrongStateTypeException(Exception):
    pass
    
