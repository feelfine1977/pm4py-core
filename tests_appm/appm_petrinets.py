import os
import pandas as pd
import pm4py
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.objects.petri_net.data_petri_nets import semantics
from pm4py.objects.petri_net.data_petri_nets.data_marking import DataMarking


def get_trans_by_name(net, name):
    ret = [x for x in net.transitions if x.name == name]
    if len(ret) == 0:
        return None
    return ret[0]


def execute_script():
    path = os.path.join( "tests_appm","input_data", "appm", "running-example.csv")
    df = pd.read_csv(path, sep=',')
    print(df)

if __name__ == "__main__":


    execute_script()