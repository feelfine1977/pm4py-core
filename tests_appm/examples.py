import pandas as pd
import os 
import pm4py
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.visualization.petri_net import visualizer as pn_vis
from pm4py.objects.bpmn.importer import importer as bpmn_importer
from pm4py.objects.conversion.bpmn import converter as bpmn_converter
from pm4py.objects.conversion.process_tree import converter as pt_converter
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.bpmn.exporter import exporter as bpmn_exporter
from pm4py.statistics.overlap.cases.log import get as case_overlap_get


PATH = os.path.join( "tests_appm","notebooks", "data")

def get_data(path=PATH, sep=';' ,file_csv="p2p_simple.csv", file_xes="p2p_simple.xes", write=False):
    path_csv = os.path.join( path, file_csv)
    path_xes = os.path.join( path, file_xes)
    df = pd.read_csv(path_csv, sep=sep)
    log_csv = pm4py.format_dataframe(df, case_id='case_id',activity_key='activity',
                             timestamp_key='timestamp')
    if write:
        pm4py.write_xes(log_csv, path_xes)
    return pm4py.read_xes(path_xes)    

def get_trans_by_name(net, name):
    ret = [x for x in net.transitions if x.name == name]
    if len(ret) == 0:
        return None
    return ret[0]


def ex_alpha_miner(log):
    log =  log
    net, i_m, f_m = alpha_miner.apply(log)

    gviz = pn_vis.apply(net, i_m, f_m,
                        parameters={pn_vis.Variants.WO_DECORATION.value.Parameters.FORMAT: "svg",
                                    pn_vis.Variants.WO_DECORATION.value.Parameters.DEBUG: False})
    pn_vis.view(gviz)

            
def ex_bpmn_import_and_to_petri_net(log, bpmn_file= 'p2p_simple.bpmn'):
    log = log
    path_bpmn = os.path.join(PATH,bpmn_file)
    bpmn_graph = bpmn_importer.apply(path_bpmn)
    net, im, fm = bpmn_converter.apply(bpmn_graph, variant=bpmn_converter.Variants.TO_PETRI_NET)
    precision_tbr = pm4py.precision_token_based_replay(log, net, im, fm)
    print("precision", precision_tbr)
    fitness_tbr = pm4py.precision_token_based_replay(log, net, im, fm)
    print("fitness", fitness_tbr)
    print(pm4py.check_soundness(net, im, fm))

def ex_bpmn_from_pt_conversion(log, bpmn_file= 'p2p_simple.bpmn'):
    log = log
    path_bpmn = os.path.join(PATH,bpmn_file)
    ptree = inductive_miner.apply_tree(log)
    bpmn = pt_converter.apply(ptree, variant=pt_converter.Variants.TO_BPMN)
    bpmn_exporter.apply(bpmn, path_bpmn)
    #pm4py.view_bpmn(bpmn, format="svg")

def ex_case_overlap_stat(log):
    # calculates the WIP statistics from the event log object.
    # The WIP statistic associates to each case the number of cases open during the lifecycle of the case
    wip = case_overlap_get.apply(log)
    print(wip)

if __name__ == "__main__":
    log = get_data()
    #ex_alpha_miner(log) #pm4py examples alpha_miner.py - shows alpha miner for log
    #ex_bpmn_from_pt_conversion(log) #convert to bpmn from petrinet
    #ex_bpmn_import_and_to_petri_net(log) #check fitness of petrinet
    ex_case_overlap_stat(log)