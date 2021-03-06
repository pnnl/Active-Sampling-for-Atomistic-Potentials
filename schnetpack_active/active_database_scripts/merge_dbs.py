from ase.db import connect
from schnetpack.data import AtomsData
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--saveas', type=str, required=True, help='file name to  save database')
parser.add_argument('--dbs', type=str, nargs="+", required=True, help='list of dbs to merge')
args = parser.parse_args()


print(f'merging {args.dbs}')


def merge_datasets(merged_dbpath, dbpaths, **mergedb_kwargs):
    merged_data = AtomsData(merged_dbpath, available_properties=['energy','forces'], **mergedb_kwargs)

    if type(dbpaths) is dict:
        names = dbpaths.keys()
        dbpaths = dbpaths.values()
    else:
        names = [dbp.split("/")[-1].split(".")[0] for dbp in dbpaths]

    partitions = {}
    offset = 0

    partition_meta = {}
    with connect(merged_data.dbpath, use_lock_file=False) as dst:
        for name, dbp in zip(names, dbpaths):
            start = offset

            if name in partitions.keys():
                count = 1
                while name + "_" + str(count) in partitions.keys():
                    count += 1
                name = name + "_" + str(count)

            with connect(dbp) as src:
                length = src.count()
                end = offset + length
                partition_meta[name] = src.metadata

                for row in src.select():
                    at = row.toatoms()
                    dst.write(at, key_value_pairs=row.key_value_pairs, data=row.data)
            partitions[name] = (start, end)
            offset += length

    metadata = {"partition_meta": partition_meta, "partitions": partitions}
    dst.metadata = metadata
    return merged_data

merge_datasets(args.saveas, args.dbs)

print(f'merged db sored at {args.saveas}')
