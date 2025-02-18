import pandas as pd
import wandb

from benchutils.utils import flatten_dict


def get_runs(entity, project):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    data = []

    for run in runs:
        config = flatten_dict(run.config)
        summary = run.summary
        name = run.name

        row = {**{'name': name}, **config, **summary}

        data.append(row)
    df = pd.DataFrame(data)
    df['_timestamp'] = pd.to_datetime(df['_timestamp'], unit='s')
    return df
