import pandas as pd
import numpy as np

def load_hits_and_truth_as_trackml(event_file, detector, mask_simhits=True):
    truth = pd.read_csv(event_file + "-truth.csv")
    truth["hit_id"] = np.array(truth.index)+1
    truth = truth.drop("index", 1)

    hits = truth[["hit_id", "geometry_id","tx","ty","tz","tt"]]
    hits = hits.rename(columns={"tx": "x", "ty": "y", "tz": "z", "tt": "t"})

    # Dummy weight
    hits["weight"] = np.ones(len(hits.index)) / len(hits.index)

    #hits[["volume_id","layer_id","module_id"]] = hits.geometry_id.map(detector.set_index('geometry_id')[["volume_id","layer_id","module_id"]].to_dict())
    hits["volume_id"] = hits.geometry_id.map(detector.set_index('geometry_id')["volume_id"].to_dict())
    hits["layer_id"] = hits.geometry_id.map(detector.set_index('geometry_id')["layer_id"].to_dict())
    hits["module_id"] = hits.geometry_id.map(detector.set_index('geometry_id')["module_id"].to_dict())

    # Apply simhit map, because digitization does not necessarily find every hit
    if mask_simhits:
        simhit_map = pd.read_csv(event_file + "-measurement-simhit-map.csv")
        simhit_mask = np.zeros(len(hits.index), dtype=bool)
        simhit_mask[simhit_map.hit_id] = True

        hits = hits[simhit_mask]
        truth = truth[simhit_mask]
    
    return hits, truth
