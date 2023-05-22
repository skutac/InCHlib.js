#coding: utf-8
from __future__ import print_function

import csv, json, copy, re, argparse, os, requests

import numpy, scipy, fastcluster, sklearn, jsmin
import scipy.cluster.hierarchy as hcluster
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from scipy import spatial

import randomcolor

try:
    import rdkit
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT = True

    FP2FNC = {
        "ecfp4": lambda rdmol: AllChem.GetMorganFingerprintAsBitVect(rdmol, radius=2, nBits=1024),
        "ecfp6": lambda rdmol: AllChem.GetMorganFingerprintAsBitVect(rdmol, radius=3, nBits=1024),
        "apfp": lambda rdmol: AllChem.GetHashedAtomPairFingerprintAsBitVect(rdmol, nBits=1024),
        "ttfp": lambda rdmol: AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(rdmol, nBits=1024),
        "maccs": lambda rdmol: AllChem.GetMACCSKeysFingerprint(rdmol),
    }
    
except Exception as e:
    RDKIT = False
    print("RDKit not found: Cheminformatic-based functionality not available...")

LINKAGES = ["single", "complete", "average", "centroid", "ward", "median", "weighted"]
RAW_LINKAGES = ["ward", "centroid"]
DISTANCES = {"numeric": ["braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "euclidean", "mahalanobis", "minkowski", "seuclidean", "sqeuclidean"],
              "binary": ["dice","hamming","jaccard","kulsinski","matching","rogerstanimoto","russellrao","sokalmichener","sokalsneath","yule"]}

class Dendrogram():
    """Class which handles the generation of cluster heatmap format of clustered data. 
    As an input it takes a Cluster instance with clustered data."""

    def __init__(self, clustering):
        self.cluster_object = clustering
        self.datatype = clustering.datatype
        self.axis = clustering.clustering_axis
        self.clustering = clustering.clustering
        self.tree = hcluster.to_tree(self.clustering)
        self.data = clustering.data
        self.data_names = clustering.data_names
        self.labels = clustering.labels
        self.header = clustering.header
        self.dendrogram = False
        self.smiles = clustering.smiles
        self.add_structures = clustering.add_structures

    def __get_cluster_heatmap__(self, write_data):
        root, nodes = hcluster.to_tree(self.clustering, rd=True)
        node_id2node = {}
        dendrogram = {"nodes":{}}

        for node in nodes:
            node_id = node.id
            if node.count == 1:
                node_id2node[node_id] = {"count":1, "distance":0}

            else:
                node_left_child = node.get_left().id
                node_right_child = node.get_right().id
                node_id2node[node_id] = {"count":node.count, "distance":round(node.dist, 3), "left_child": node_left_child, "right_child": node_right_child}

        for n, node in node_id2node.items():
            if node["count"] != 1:
                node_id2node[node["left_child"]]["parent"] = n
                node_id2node[node["right_child"]]["parent"] = n

        for n, node in node_id2node.items():

            if node["count"] == 1:
                data = self.data[n]
                node["objects"] = [self.data_names[n]]
                node["label"] = self.labels[n]

                if self.add_structures:
                    node["structure"] = self.smiles[n]

                if node_id2node[node["parent"]]["left_child"] == n:
                    node_id2node[node["parent"]]["left_child"] = n
                else:
                    node_id2node[node["parent"]]["right_child"] = n

                if not write_data:
                    data = []

                node["features"] = data
                dendrogram["nodes"][n] = node

        for n in node_id2node:
             if node_id2node[n]["count"] != 1:
                dendrogram["nodes"][n] = node_id2node[n]

        return dendrogram

    def __get_column_dendrogram__(self):
        root, nodes = hcluster.to_tree(self.cluster_object.column_clustering, rd=True)
        node_id2node = {}
        dendrogram = {"nodes":{}}

        for node in nodes:
            node_id = node.id
            if node.count == 1:
                node_id2node[node_id] = {"count":1, "distance":0}

            else:
                node_left_child = node.get_left().id
                node_right_child = node.get_right().id
                node_id2node[node_id] = {"count":node.count, "distance":round(node.dist, 3), "left_child": node_left_child, "right_child": node_right_child}

        for n in node_id2node:
            node = node_id2node[n]
            if node["count"] != 1:
                node_id2node[node["left_child"]]["parent"] = n
                node_id2node[node["right_child"]]["parent"] = n

        for n in node_id2node:
             if not n in dendrogram["nodes"]:
                dendrogram["nodes"][n] = node_id2node[n]

        return dendrogram

    def __get_leaves_for_node__(self, nodeid):
        nodes = [nodeid]
        leaves = []

        while len(nodes):
            for nodeid in nodes:
                node = self.dendrogram["data"]["nodes"][nodeid]
                if node["count"] > 1:
                    nodes.extend([node["left_child"], node["right_child"]])

                else:
                    leaves.append(nodeid)

                nodes.remove(nodeid)

        return leaves


    def create_cluster_heatmap(self, compress=False, compressed_value="median", write_data=True):
        """Creates cluster heatmap representation in inchlib format. By setting compress parameter to True you can
        cut the dendrogram in a distance to decrease the row size of the heatmap to specified count. 
        When compressing the type of the resulted value of merged rows is given by the compressed_value parameter (median, mean).
        When the metadata are nominal (text values) the most frequent is the result after compression.
        By setting write_data to False the data features won't be present in the resulting format."""
        self.dendrogram = {"data": self.__get_cluster_heatmap__(write_data)}

        self.compress = compress
        self.compressed_value = compressed_value
        self.compress_cluster_threshold = 0
        if self.compress and self.compress >= 0:
            self.compress_cluster_threshold = self.__get_distance_threshold__(compress)
            print("Distance threshold for compression:", self.compress_cluster_threshold)
            if self.compress_cluster_threshold >= 0:
                self.__compress_data__()
        else:
            self.compress = False

        if self.header and write_data:
            self.dendrogram["data"]["feature_names"] = [h for h in self.header]
        elif self.header and not write_data:
            self.dendrogram["data"]["feature_names"] = []
        
        if self.axis == "both" and len(self.cluster_object.column_clustering):
            column_dendrogram = hcluster.to_tree(self.cluster_object.column_clustering)            
            self.dendrogram["column_dendrogram"] = self.__get_column_dendrogram__()

    def color_clusters(self, cluster_count):
        """Color given number of clusters based on a dendrogram cut
        
        Arguments:
            cluster_count {[int]} -- numer of clusters
        """
        if cluster_count > 1:
            self.cluster_distance_threshold = self.__get_distance_threshold__(cluster_count)
            rand_color = randomcolor.RandomColor()
            
            to_color = []
            for nodeid, node in self.dendrogram["data"]["nodes"].items():
                if node["distance"] < self.cluster_distance_threshold and self.dendrogram["data"]["nodes"].get(node["parent"], {"distance": 0})["distance"] > self.cluster_distance_threshold:
                    to_color.append(nodeid)

            colors = rand_color.generate(count=len(to_color))
            for i, nodeid in enumerate(to_color):
                node_leaves = self.__get_leaves_for_node__(nodeid)
                self.dendrogram["data"]["nodes"][nodeid]["color"] = colors[i]

                for lid in node_leaves:
                    self.dendrogram["data"]["nodes"][lid]["cluster"] = i

    def __compress_data__(self):
        nodes = {}
        to_remove = set()

        compressed_value2fnc = {
            "median": lambda values: [round(numpy.median([v for v in value if v is not None]), 3) if len([v for v in value if v is not None]) else None for value in values],
            "mean": lambda values: [round(numpy.average([v for v in value if v is not None]), 3) if len([v for v in value if v is not None]) else None for value in values],
        }
        
        for n in self.dendrogram["data"]["nodes"]:
            node = self.dendrogram["data"]["nodes"][n]

            if node["count"] == 1:
                objects = node["objects"]
                data = node["features"]
                node_id = n

                while self.dendrogram["data"]["nodes"][node["parent"]]["distance"] <= self.compress_cluster_threshold:
                    to_remove.add(node_id)
                    node_id = node["parent"]
                    node = self.dendrogram["data"]["nodes"][node_id]

                if node["count"] != 1:

                    if not "objects" in self.dendrogram["data"]["nodes"][node_id]:
                        self.dendrogram["data"]["nodes"][node_id]["objects"] = []
                        self.dendrogram["data"]["nodes"][node_id]["features"] = []
                    
                    self.dendrogram["data"]["nodes"][node_id]["objects"].extend(objects)

                    if data:
                        self.dendrogram["data"]["nodes"][node_id]["features"].append(data)

        for node in to_remove:
            self.dendrogram["data"]["nodes"].pop(node)

        for k in self.dendrogram["data"]["nodes"]:
            node = self.dendrogram["data"]["nodes"][k]
            if "objects" in node and node["count"] != 1:
                self.dendrogram["data"]["nodes"][k]["distance"] = 0
                self.dendrogram["data"]["nodes"][k]["count"] = 1
                self.dendrogram["data"]["nodes"][k].pop("left_child")
                self.dendrogram["data"]["nodes"][k].pop("right_child")
                rows = zip(*self.dendrogram["data"]["nodes"][k]["features"])
                self.dendrogram["data"]["nodes"][k]["features"] = compressed_value2fnc[self.compressed_value](rows)

        self.__adjust_node_counts__()

    def __adjust_node_counts__(self):
        leaves = []

        for n in self.dendrogram["data"]["nodes"]:
            if self.dendrogram["data"]["nodes"][n]["count"] > 1:
                self.dendrogram["data"]["nodes"][n]["count"] = 0
            else:
                leaves.append(n)

        for n in leaves:
            node = self.dendrogram["data"]["nodes"][n]
            parent_id = node["parent"]

            while parent_id:
                node = self.dendrogram["data"]["nodes"][parent_id]
                self.dendrogram["data"]["nodes"][parent_id]["count"] += 1
                parent_id = False
                if "parent" in node:
                    parent_id = node["parent"]

    def __get_distance_threshold__(self, cluster_count):
        print("Calculating distance threshold...")
        if cluster_count >= self.tree.count:
            return -1
        
        i = 0
        count = cluster_count + 1
        test_step = self.tree.dist/2

        while test_step >= 0.1:
            count = len(set([c for c in hcluster.fcluster(self.clustering, i, "distance")]))
            if count < cluster_count:
                if i == 0:
                    return 0
                i = i - test_step
                test_step = test_step/2
            elif count == cluster_count:
                return i
            else:
                i += test_step

        return i+test_step*2

    def export_cluster_heatmap_as_json(self, filename=None, minify=False, dump=True):
        """Returns cluster heatmap in a JSON format or exports it to the file specified by the filename parameter."""
        dendrogram = self.dendrogram
        if minify:
            dendrogram_json = json.dumps(dendrogram)
            dendrogram_json = self.__minify_data(dendrogram_json)
        elif dump:
            dendrogram_json = json.dumps(dendrogram, indent=4)
        else:
            dendrogram_json = dendrogram
        
        if filename:
            output = open(filename, "w")
            output.write(dendrogram_json)
        return dendrogram_json

    def __minify_data(self, data):
        return jsmin.jsmin(str(data))

    def export_cluster_heatmap_as_html(self, htmldir="."):
        """Export simple HTML page with embedded cluster heatmap and dependencies to given directory."""
        if not os.path.exists(htmldir):
            os.makedirs(htmldir)
        dendrogram_json = json.dumps(self.dendrogram, indent=4)
        template = """<html>
        <head>
            <script src="jquery-2.0.3.min.js"></script>
            <script src="kinetic-v5.1.0.min.js"></script>
            <script src="inchlib-1.1.0.js"></script>
            <script>
            $(document).ready(function() {{
                var data = {};
                var inchlib = new InCHlib({{
                    target: "inchlib",
                    max_height: 1200,
                    width: 1000,
                }});
                inchlib.read_data(data);
                inchlib.draw();
            }});
            </script>
        </head>

        <body>
            <div id="inchlib"></div>
        </body>
        </html>""".format(dendrogram_json)

        lib2url = {
            "inchlib-1.1.0.js": "https://openscreen.cz/software/inchlib/static/js/inchlib-1.1.0.js",
            "jquery-2.0.3.min.js": "https://openscreen.cz/software/inchlib/static/js/jquery-2.0.3.min.js",
            "kinetic-v5.1.0.min.js": "https://openscreen.cz/software/inchlib/static/js/kinetic-v5.1.0.min.js"
        }
        
        for lib, url in lib2url.items():
            try:
                source = requests.get(url)
                source_html = source.content

                with open(os.path.join(htmldir, lib), "w") as output:
                    output.write(source_html)
            
            except Exception as e:
                raise Exception("\nCan't download file {}.\nPlease check your internet connection and try again.\nIf the error persists there can be something wrong with the InCHlib server.\n".format(url))

        with open(os.path.join(htmldir, "inchlib.html"), "w") as output:
            output.write(template)

    def add_metadata_from_file(self, metadata_file, delimiter, header=True, metadata_compressed_value="median"):
        """Adds metadata from csv file.
        Metadata_compressed_value specifies the resulted value when the data are compressed (median/mean/frequency)"""
        self.metadata_compressed_value = metadata_compressed_value
        self.metadata, self.metadata_header = self.__read_metadata_file__(metadata_file, delimiter, header)
        self.__connect_metadata_to_data__()

    def add_metadata(self, metadata, header=True, metadata_compressed_value="median"):
        """Adds metadata in a form of list of lists (tuples).
        Metadata_compressed_value specifies the resulted value when the data are compressed (median/mean/frequency)"""
        self.metadata_compressed_value = metadata_compressed_value
        self.metadata, self.metadata_header = self.__read_metadata__(metadata, header)
        self.__connect_metadata_to_data__()

    def __connect_metadata_to_data__(self):
        print("Adding metadata: {} rows".format(len(self.metadata)))
        self.dendrogram["metadata"] = {}

        if self.metadata_header:
            self.dendrogram["metadata"]["feature_names"] = self.metadata_header

        self.dendrogram["metadata"]["nodes"] = self.__connect_additional_data_to_data__(self.metadata, self.metadata_compressed_value)

    def __read_metadata__(self, metadata, header):
        metadata_header = []
        rows = metadata
        metadata = {}
        data_start = 0

        if header:
            metadata_header = rows[0][1:]
            data_start = 1
        
        for row in rows[data_start:]:
            metadata[str(row[0])] = [r for r in row[1:]]

        return metadata, metadata_header

        
    def __read_metadata_file__(self, metadata_file, delimiter, header):
        csv_reader = csv.reader(open(metadata_file, "r"), delimiter=delimiter)
        metadata_header = []
        rows = [row for row in csv_reader]
        metadata = {}
        data_start = 0

        if header:
            metadata_header = rows[0][1:]
            data_start = 1
        
        for row in rows[data_start:]:
            metadata_id = str(row[0])
            metadata[metadata_id] = [r for r in row[1:]]

        return metadata, metadata_header

    def add_column_metadata(self, column_metadata, header=True):
        """Adds column metadata in a form of list of lists (tuples). 
        Column metadata doesn't have header row, first item in each row is used as label instead"""
        if header:
            self.column_metadata = [r[1:] for r in column_metadata]
            self.column_metadata_header = [r[0] for r in column_metadata]
        else:
            self.column_metadata = [r for r in column_metadata]
            self.column_metadata_header = False

        self.__check_column_metadata_length__()
        self.__add_column_metadata_to_data__()

    def add_column_metadata_from_file(self, column_metadata_file, delimiter=",", header=True):
        """Adds column metadata from csv file. Column metadata doesn't have header."""
        csv_reader = csv.reader(open(column_metadata_file, "r"), delimiter=delimiter)
        column_metadata = [row for row in csv_reader]
        self.add_column_metadata(column_metadata, header)

    def __check_column_metadata_length__(self):
        features_length = len(self.data[0])
        for row in self.column_metadata:
            if features_length != len(row):
                raise Exception("Column metadata length and features length must be the same.")

    def __add_column_metadata_to_data__(self):
        if self.cluster_object.clustering_axis == "both":
            self.column_data = self.cluster_object.__reorder_data__(self.column_metadata, self.cluster_object.data_order)
        self.dendrogram["column_metadata"] = {"features":self.column_metadata}
        if self.column_metadata_header:
            self.dendrogram["column_metadata"]["feature_names"] = self.column_metadata_header

    def add_alternative_data_from_file(self, alternative_data_file, delimiter, header, alternative_data_compressed_value):
        """Adds alternative_data from csv file."""
        self.alternative_data_compressed_value = alternative_data_compressed_value
        self.add_alternative_data(self.__read_alternative_data_file__(alternative_data_file, delimiter), header, alternative_data_compressed_value)

    def add_alternative_data(self, alternative_data, header, alternative_data_compressed_value):
        """Adds alternative data in a form of list of lists (tuples)."""
        self.alternative_data_compressed_value = alternative_data_compressed_value

        if self.cluster_object.clustering_axis == "both":
            alternative_data = self.__reorder_alternative_data__(alternative_data)

        self.dendrogram["alternative_data"] = {}
        self.alternative_data_header = False
        
        if header:
            self.alternative_data_header = alternative_data[0][1:]
            self.dendrogram["alternative_data"]["feature_names"] = self.alternative_data_header
            alternative_data = alternative_data[1:]

        self.alternative_data = self.__read_alternative_data__(alternative_data)

        print("Adding alternative data: {} rows".format(len(self.alternative_data)))
        self.dendrogram["alternative_data"]["nodes"] = self.__connect_additional_data_to_data__(self.alternative_data, self.alternative_data_compressed_value)

    def __reorder_alternative_data__(self, alternative_data):
        alt_data_without_id = [r[1:] for r in alternative_data]
        reordered_data = self.cluster_object.__reorder_data__(alt_data_without_id, self.cluster_object.data_order)
        rows = []
        for i, r in enumerate(alternative_data):
            row = [r[0]]
            row.extend(reordered_data[i])
            rows.append(row)
        return rows

    def __read_alternative_data_file__(self, alternative_data_file, delimiter):
        csv_reader = csv.reader(open(alternative_data_file, "r"), delimiter=delimiter)
        return [r for r in csv_reader]

    def __read_alternative_data__(self, alternative_data):
        return {str(r[0]):r[1:] for r in alternative_data}

    def __connect_additional_data_to_data__(self, additional_data, compressed_value):
        if len(set(additional_data.keys()) & set(self.data_names)) == 0:
            print("No data objects correspond with the clustered data according to their IDs. No additional data added.")
            return

        if not self.dendrogram:
            raise Exception("You must create dendrogram before adding data to it.")

        node2additional_data = {}

        leaves = {n:node for n, node in self.dendrogram["data"]["nodes"].items() if node["count"] == 1}

        if not self.compress:
            for leaf_id, leaf in leaves.items():
                try:
                    node2additional_data[leaf_id] = additional_data[leaf["objects"][0]]
                except Exception as e:
                    continue
        else:
            compressed_value2fnc = {
                "median": lambda values: round(numpy.median(col), 3),
                "mean": lambda values: round(numpy.average(col), 3),
                "frequency": lambda values: self.__get_most_frequent__(col)
            }

            for leaf in leaves:
                objects = []
                for item in leaves[leaf]["objects"]:
                    try:
                        objects.append(additional_data[item])
                    except Exception as e:
                        continue

                cols = zip(*objects)
                row = []
                cols = [list(c) for c in cols]

                for col in cols:
                    if compressed_value in compressed_value2fnc:
                        try:
                            col = [float(c) for c in col]
                            value = compressed_value2fnc[compressed_value](col)
                        except ValueError:
                            value = compressed_value2fnc["frequency"](col)
                    
                    else:
                        raise Exception("Unkown type of metadata_compressed_value: {}. Possible values are: median, mean, frequency.".format(self.metadata_compressed_value))
        
                    row.append(value)

                node2additional_data[leaf] = row

        return node2additional_data

    def __get_most_frequent__(self, col):
        freq2val = {col.count(v):v for v in set(col)}
        value = freq2val[max(freq2val.keys())]
        return value

class Cluster():
    """Class for data clustering"""

    def __init__(self):
        self.write_original = False

    def read_csv(self, filename, delimiter=",", header=False, missing_values=False, datatype="numeric", compound_structure_field=False, add_structures=False, label_field=False):
        """Reads data from the CSV file"""
        self.filename = filename
        csv_reader = csv.reader(open(self.filename, "r"), delimiter=delimiter)
        rows = [row for row in csv_reader]
        self.read_data(rows, header, missing_values, datatype, compound_structure_field, add_structures, label_field)

    def read_data(self, rows, header=False, missing_values=False, datatype="numeric", compound_structure_field=False, add_structures=False, label_field=False):
        """Reads data in a form of list of lists (tuples)"""
        self.datatype = datatype
        self.missing_values = missing_values
        self.header = header
        self.compound_structure_field = compound_structure_field
        self.label_field = label_field
        self.labels = False
        self.smiles = False
        self.rdmols = False
        self.fpobjs = False
        self.add_structures = add_structures

        data_start = 0

        if self.header and self.compound_structure_field and self.compound_structure_field in rows[0]:
            print("Reading compound structures...")
            csf_index = rows[0].index(self.compound_structure_field)
            self.smiles = [row[csf_index] for row in rows[1:]]

            if RDKIT:
                self.rdmols = [Chem.MolFromSmiles(row[csf_index]) for row in rows[1:]]
                self.fpobjs = [FP2FNC["ecfp4"](rdmol) for rdmol in self.rdmols]
            
            for row in rows:
                row.pop(csf_index)

        if self.header and self.label_field:
            label_index = rows[0].index(self.label_field)
            self.labels = [row[label_index] for i, row in enumerate(rows[1:])]
            
            for row in rows:
                row.pop(label_index)

        if self.header:
            self.header = rows[0][1:]
            data_start = 1
        
        self.data_names = [str(row[0]) for row in rows[data_start:]]
        self.data = numpy.array([row[1:] for row in rows[data_start:]])
        self.original_data = copy.deepcopy(self.data)

        if not self.missing_values is False:
            self.data, self.missing_values_indexes = self.__impute_missing_values__(self.data)
            self.original_data = self.__return_missing_values__(copy.deepcopy(self.data), self.missing_values_indexes)

        self.original_data = [[float(val) if not val is None else None for val in r] for r in self.original_data]
        self.data = [[float(val) if not val is None else None for val in r] for r in self.data]
        
    def __impute_missing_values__(self, data):
        datatype2impute = {"numeric": {"strategy":"mean", 
                                        "value": lambda value: round(float(value), 3)}, 
                           "binary": {"strategy":"most_frequent", 
                                      "value": lambda value: int(value)}
                           }

        if not self.datatype in DISTANCES:
            raise Exception("".join(["You can choose only from data types: ", ", ".join(DISTANCES.keys())]))

        missing_values_indexes = []
        
        for i, row in enumerate(self.data):
            missing_values_indexes.append([j for j, v in enumerate(row) if v in self.missing_values])

            for j, value in enumerate(row):
                if value in self.missing_values:
                    data[i][j] = numpy.nan

        imputer = SimpleImputer(missing_values=numpy.nan, strategy=datatype2impute[self.datatype]["strategy"], keep_empty_features=True)
        imputed_data = [list(row) for row in imputer.fit_transform(self.data)]
        imputed_data = [[datatype2impute[self.datatype]["value"](value) for value in row] for row in imputed_data]
        return imputed_data, missing_values_indexes
        
    def normalize_data(self, feature_range=(0,1), write_original=False):
        """Normalizes data to a scale from 0 to 1. When write_original is set to True, 
        the normalized data will be clustered, but original data will be written to the heatmap."""
        self.write_original = write_original
        min_max_scaler = MinMaxScaler(feature_range)
        self.data = min_max_scaler.fit_transform(self.data)
        self.data = [[round(v, 3) for v in row] for row in self.data]

    def cluster_data(self, row_distance="euclidean", row_linkage="single", axis="row", column_distance="euclidean", column_linkage="ward"):
        """Performs clustering according to the given parameters.
        @datatype - numeric/binary
        @row_distance/column_distance - see. DISTANCES variable
        @row_linkage/column_linkage - see. LINKAGES variable
        @axis - row/both
        """
        print("Clustering rows:", row_distance, row_linkage)
        self.clustering_axis = axis
        row_linkage = str(row_linkage)
        
        if row_linkage in RAW_LINKAGES:
            self.clustering = fastcluster.linkage(self.data, method=row_linkage, metric=row_distance)

        else:
            self.distance_vector = fastcluster.pdist(self.data, row_distance)

            if self.datatype == "numeric" and not row_distance in DISTANCES[self.datatype]:
                raise Exception("".join(["When clustering numeric data you must choose from these distance measures: ", ", ".join(DISTANCES[self.datatype])]))
            elif (self.datatype == "binary" or self.datatype == "nominal") and not row_distance in DISTANCES[self.datatype]:
                raise Exception("".join(["When clustering binary or nominal data you must choose from these distance measures: ", ", ".join(DISTANCES[self.datatype])]))

            self.clustering = fastcluster.linkage(self.distance_vector, method=str(row_linkage))


        if not self.missing_values is False:
            self.data = self.__return_missing_values__(self.data, self.missing_values_indexes)
        self.column_clustering = []

        if axis == "both" and len(self.data[0]) > 2:
            print("Clustering columns:", column_distance, column_linkage)
            self.__cluster_columns__(column_distance, column_linkage)
        
        if self.write_original or self.datatype == "nominal":
            self.data = self.original_data

    def __return_missing_values__(self, data, missing_values_indexes):
        for i, indexes in enumerate(missing_values_indexes):
            if indexes:
                print(data[i])
                # print(indexes)
                for index in indexes:
                    data[i][index] = None
        return data

    def __cluster_columns__(self, column_distance, column_linkage):
        self.data = [list(col) for col in zip(*self.data)]
        if not self.missing_values is False:
            self.data, missing_values_indexes = self.__impute_missing_values__(self.data)
        
        self.column_clustering = fastcluster.linkage(self.data, method=column_linkage, metric=column_distance)
        self.data_order = hcluster.leaves_list(self.column_clustering)

        if not self.missing_values is False:
            self.data = self.__return_missing_values__(self.data, missing_values_indexes)
        
        self.data = list(zip(*self.data))
        self.data = self.__reorder_data__(self.data, self.data_order)
        self.original_data = self.__reorder_data__(self.original_data, self.data_order)
        if self.header:
            self.header = self.__reorder_data__([self.header], self.data_order)[0]

    def __reorder_data__(self, data, order):
        for i in range(len(data)):
            reordered_data = []
            for j in order:
                reordered_data.append(data[i][j])
            reordered_data.reverse()
            data[i] = reordered_data

        return data

def _process_(arguments):
    c = Cluster()
    c.read_csv(
        filename=arguments.data_file, 
        delimiter=arguments.data_delimiter, 
        header=arguments.data_header, 
        missing_values=arguments.missing_values, 
        datatype=arguments.datatype,
        compound_structure_field=arguments.compound_structure_field,
        add_structures=arguments.add_structures,
        label_field=arguments.label_field
    )
    
    if arguments.normalize:
        c.normalize_data(feature_range=(0,1), write_original=arguments.write_original)

    c.cluster_data(row_distance=arguments.row_distance, row_linkage=arguments.row_linkage, axis=arguments.axis, column_distance=arguments.column_distance, column_linkage=arguments.column_linkage)

    d = Dendrogram(c)
    d.create_cluster_heatmap(compress=arguments.compress, compressed_value=arguments.compressed_value, write_data=not arguments.dont_write_data)

    if arguments.color_clusters > 1:
        d.color_clusters(cluster_count=arguments.color_clusters)
    
    if arguments.metadata:
        d.add_metadata_from_file(metadata_file=arguments.metadata, delimiter=arguments.metadata_delimiter, header=arguments.metadata_header, metadata_compressed_value=arguments.metadata_compressed_value)
    
    if arguments.column_metadata:
        d.add_column_metadata_from_file(column_metadata_file=arguments.column_metadata, delimiter=arguments.column_metadata_delimiter, header=arguments.column_metadata_header)

    if arguments.alternative_data:
        d.add_alternative_data_from_file(alternative_data_file=arguments.alternative_data, delimiter=arguments.alternative_data_delimiter, header=arguments.alternative_data_header, alternative_data_compressed_value=arguments.alternative_data_compressed_value)
    
    if arguments.output_file or arguments.html_dir:
        if arguments.output_file:
            d.export_cluster_heatmap_as_json(arguments.output_file, minify=arguments.minify, dump=arguments.json_dump)
        else:
            d.export_cluster_heatmap_as_html(arguments.html_dir)
    else:
        print(d.export_cluster_heatmap_as_json(filename=None, minify=arguments.minify, dump=arguments.json_dump))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("data_file", type=str, help="csv(text) data file with delimited values")
    parser.add_argument("-o", "--output_file", type=str, help="the name of output file")
    parser.add_argument("-html", "--html_dir", type=str, help="the directory to store HTML page with dependencies")
    parser.add_argument("-rd", "--row_distance", type=str, default="euclidean", help="set the distance to use for clustering rows")
    parser.add_argument("-rl", "--row_linkage", type=str, default="ward", help="set the linkage to use for clustering rows")
    parser.add_argument("-cd", "--column_distance", type=str, default="euclidean", help="set the distance to use for clustering columns (only when clustering by both axis -a parameter)")
    parser.add_argument("-cl", "--column_linkage", type=str, default="ward", help="set the linkage to use for clustering columns (only when clustering by both axis -a parameter)")
    parser.add_argument("-a", "--axis", type=str, default="row", help="define clustering axis (row/both)")
    parser.add_argument("-dt", "--datatype", type=str, default="numeric", help="specify the type of the data (numeric/binary)")
    parser.add_argument("-dd", "--data_delimiter", type=str, default=",", help="delimiter of values in data file")
    parser.add_argument("-m", "--metadata", type=str, default=None, help="csv(text) metadata file with delimited values")
    parser.add_argument("-md", "--metadata_delimiter", type=str, default=",", help="delimiter of values in metadata file")
    parser.add_argument("-dh", "--data_header", default=False, help="whether the first row of data file is a header", action="store_true")
    parser.add_argument("-mh", "--metadata_header", default=False, help="whether the first row of metadata file is a header", action="store_true")
    parser.add_argument("-c", "--compress", type=int, default=0, help="compress the data to contain maximum of specified count of rows")
    parser.add_argument("-cv", "--compressed_value", type=str, default="median", help="the resulted value from merged rows when the data are compressed (median/mean/frequency)")
    parser.add_argument("-mcv", "--metadata_compressed_value", type=str, default="median", help="the resulted value from merged rows of metadata when the data are compressed (median/mean/frequency)")
    parser.add_argument("-dwd", "--dont_write_data", default=False, help="don't write clustered data to the inchlib data format", action="store_true")
    parser.add_argument("-n", "--normalize", default=False, help="normalize data to [0, 1] range", action="store_true")
    parser.add_argument("-wo", "--write_original", default=False, help="cluster normalized data but write the original ones to the heatmap", action="store_true")
    parser.add_argument("-cm", "--column_metadata", type=str, default=None, help="csv(text) metadata file with delimited values without header")
    parser.add_argument("-cmd", "--column_metadata_delimiter", type=str, default=",", help="delimiter of values in column metadata file")
    parser.add_argument("-cmh", "--column_metadata_header", default=False, help="whether the first column of the column metadata is the row label ('header')", action="store_true")
    parser.add_argument("-mv", "--missing_values", type=str, nargs="+", default=False, help="define the string representating missing values in the data")
    parser.add_argument("-ad", "--alternative_data", type=str, default=None, help="csv(text) alternative data file with delimited values")
    parser.add_argument("-adh", "--alternative_data_header", default=False, help="whether the first row of alternative data file is a header", action="store_true")
    parser.add_argument("-add", "--alternative_data_delimiter", type=str, default=",", help="delimiter of values in alternative data file")
    parser.add_argument("-adcv", "--alternative_data_compressed_value", type=str, default="median", help="the resulted value from merged rows of alternative data when the data are compressed (median/mean/frequency)")
    parser.add_argument("-min", "--minify", default=False, help="minify the InCHlib format", action="store_true")
    parser.add_argument("-dump", "--json_dump", default=True, help="dump the InCHlib format as JSON", action="store_true")
    parser.add_argument("-cc", "--color_clusters", type=int, default=0, help="color defined number of clusters")

    parser.add_argument("-csf", "--compound_structure_field", type=str, default=False, help="the name of a column with a compound structure")
    parser.add_argument("-as", "--add_structures", default=False, help="add structure smiles to the output json format", action="store_true")
    parser.add_argument("-lf", "--label_field", type=str, default=False, help="set a label field name in case it is in the data file")
    
    args = parser.parse_args()
    _process_(args)

