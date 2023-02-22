import asyncio
import json

import hist
import matplotlib as mpl
import matplotlib.pyplot as plt
import uproot

from func_adl_servicex import ServiceXSourceUpROOT
from func_adl import ObjectStream
from coffea.processor import servicex
from servicex import ServiceXDataset


def get_client(af="coffea_casa"):
    if af == "coffea_casa":
        from dask.distributed import Client

        client = Client("tls://localhost:8786")

    elif af == "EAF":
        from lpcdaskgateway import LPCGateway

        gateway = LPCGateway()
        cluster = gateway.new_cluster()
        cluster.scale(10)
        print("Please allow up to 60 seconds for HTCondor worker jobs to start")
        print(f"Cluster dashboard: {str(cluster.dashboard_link)}")

        client = cluster.get_client()

    elif af == "local":
        from dask.distributed import Client

        client = Client()

    else:
        raise NotImplementedError(f"unknown analysis facility: {af}")

    return client


def construct_fileset(n_files_max_per_sample, use_xcache=False, af_name="", ntuples_json="nanoaod_inputs.json"):
    # using https://atlas-groupdata.web.cern.ch/atlas-groupdata/dev/AnalysisTop/TopDataPreparation/XSection-MC15-13TeV.data
    # for reference
    # x-secs are in pb
    xsec_info = {
        "ttbar": 396.87 + 332.97, # nonallhad + allhad, keep same x-sec for all
        "single_top_s_chan": 2.0268 + 1.2676,
        "single_top_t_chan": (36.993 + 22.175)/0.252,  # scale from lepton filter to inclusive
        "single_top_tW": 37.936 + 37.906,
        "wjets": 61457 * 0.252,  # e/mu+nu final states
        "data": None
    }

    # list of files
    with open(ntuples_json) as f:
        file_info = json.load(f)

    # process into "fileset" summarizing all info
    fileset = {}
    for process in file_info.keys():
        if process == "data":
            continue  # skip data

        for variation in file_info[process].keys():
            file_list = file_info[process][variation]["files"]
            if n_files_max_per_sample != -1:
                file_list = file_list[:n_files_max_per_sample]  # use partial set of samples

            file_paths = [f["path"] for f in file_list]
            if use_xcache:
                file_paths = [f.replace("https://xrootd-local.unl.edu:1094", "root://red-xcache1.unl.edu") for f in file_paths]
            if af_name == "ssl-dev":
                # point to local files on /data
                file_paths = [f.replace("https://xrootd-local.unl.edu:1094//store/user/", "/data/alheld/") for f in file_paths]
            nevts_total = sum([f["nevts"] for f in file_list])
            metadata = {"process": process, "variation": variation, "nevts": nevts_total, "xsec": xsec_info[process]}
            fileset.update({f"{process}__{variation}": {"files": file_paths, "metadata": metadata}})

    return fileset


# modified from generate_config in https://github.com/triton-inference-server/fil_backend/blob/main/qa/L0_e2e/generate_example_model.py
def generate_triton_config(model_name, 
                           num_features, 
                           model_format='xgboost',
                           num_classes=2, 
                           instance_kind='gpu', 
                           predict_proba=False,
                           task='classification',
                           threshold=0.5,
                           max_batch_size=500_000,
                           storage_type="AUTO"):
    
    """Return a string with the full Triton config.pbtxt for this model
    """
    if instance_kind == 'gpu':
        instance_kind = 'KIND_GPU'
    elif instance_kind == 'cpu':
        instance_kind = 'KIND_CPU'
    else:
        raise ValueError("instance_kind must be either 'gpu' or 'cpu'")
        
    if predict_proba:
        output_dim = num_classes
    else:
        output_dim = 1
        
    predict_proba = str(bool(predict_proba)).lower()
    output_class = str(task == 'classification').lower()
    
    if model_format == 'pickle':
        model_format = 'treelite_checkpoint'

    # Add treeshap output to xgboost_shap model
    treeshap_output_dim = num_classes if num_classes > 2 else 1
    if treeshap_output_dim == 1:
        treeshap_output_str = f"{num_features + 1}"
    else:
        treeshap_output_str = f"{treeshap_output_dim}, {num_features + 1}"
    treeshap_output = ""
    if model_name == 'xgboost_shap':
        treeshap_output = f"""
        ,{{
            name: "treeshap_output"
            data_type: TYPE_FP32
            dims: [ {treeshap_output_str} ]
        }}
        """
        
    return f"""name: "{model_name}"
backend: "fil"
max_batch_size: {max_batch_size}
input [
 {{
    name: "input__0"
    data_type: TYPE_FP32
    dims: [ {num_features} ]
  }}
]
output [
 {{
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ {output_dim} ]
  }}
 {treeshap_output}
]
instance_group [{{ kind: {instance_kind} }}]
parameters [
  {{
    key: "model_type"
    value: {{ string_value: "{model_format}" }}
  }},
  {{
    key: "predict_proba"
    value: {{ string_value: "{predict_proba}" }}
  }},
  {{
    key: "output_class"
    value: {{ string_value: "{output_class}" }}
  }},
  {{
    key: "threshold"
    value: {{ string_value: "{threshold}" }}
  }},
  {{
    key: "algo"
    value: {{ string_value: "ALGO_AUTO" }}
  }},
  {{
    key: "storage_type"
    value: {{ string_value: "{storage_type}" }}
  }},
  {{
    key: "blocks_per_sm"
    value: {{ string_value: "0" }}
  }}
]

dynamic_batching {{ }}"""
    