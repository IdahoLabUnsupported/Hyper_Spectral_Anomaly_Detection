# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
import pandas as pd
import numpy as np
import datetime
import sys
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer as vecorizer
import re
from loguru import logger
from loguru_config import LoguruConfig
import os
import json

path = os.path.dirname(os.path.realpath(__file__))


# Apply the configuration to the logger
# LoguruConfig.load(
#     f"{path}/loguru_config.json"
# )
def reshape_a_b_c__ab_c(array: np.ndarray = None) -> np.ndarray:
    """Takes an a x b x l vector data set and ravels it to an (ab) x l data set."""
    for row in range(len(array)):
        try:
            data = np.append(data, array[row, :, :], axis=0)
        except:
            data = array[row, :, :]
    return data


def edit_loguru_config(
    path,
    base_directory,
    unique_id_str,
):
    with open(f"{path}/loguru_config.json") as config:
        log_config = json.load(config)
        loguru_log_path = f"{base_directory}{unique_id_str}/logs/HSA_loguru.log"
        log_config["handlers"][0]["sink"] = loguru_log_path
    print(loguru_log_path)
    with open(f"{path}/loguru_config.json", "w") as config:
        json.dump(log_config, config)
        config.close()


def set_directories(
    log_directory: str,
    results_directory: str,
):
    """Checks or makes log and results directory and updates the loguru config file to write to the log_directory. Returns directories as str."""

    os.makedirs(log_directory, exist_ok=True)
    os.makedirs(results_directory, exist_ok=True)

    LoguruConfig.load(f"{path}/loguru_config.json")
    logger.trace("HSA Model directories set, and Loguru logger configured.")

    return log_directory, results_directory


class filter_prior_predictions:
    def __init__(
        self,
        results_path: str = None,
        lookback_days: int = 0,
        static_key: str = None,
        todays_df=None,
        run_date=None,
    ):
        """INPUTS
        results path --> (str) path to results.json
        lookback_days --> (int) number of days that results should not be repeated in
        static_key --> (str) key from results_df that will be static across comparison dates
        *results_file_pattern --> f-string with d as datetime var ex: f"Results_{'{:04d}'.format(d.year)}-{'{:02d}'.format(d.month)}-{'{:02d}'.format(d.day)}.json"

        OUTPUTS
        todays_results --> (pd.DataFrame()) results from resent run with repeated results removed.
        """
        self.results_path = results_path
        self.lookback_days = lookback_days
        self.static_key = static_key
        self.todays_df = todays_df
        self.run_date = run_date

    def get_results_lookback(self):
        """INPUTS
        results path --> (str) path to results.json
        lookback_days --> (int) number of days that results should not be repeated in

        OUTPUTS
        self.prior_json --> (list) list of results.json to filter against
        time_range_results --> (pd.DataFrame()) set of prior predictions over lookback_days
        self.null_results_list--> (list) list of results whose dates are not present in the results_path
        """
        today = self.run_date
        date_list = [
            today - datetime.timedelta(days=x) for x in range(1, 1 + self.lookback_days)
        ]
        self.prior_json = []
        for d in date_list:
            self.prior_json.append(
                f"Results_{'{:04d}'.format(d.year)}-{'{:02d}'.format(d.month)}-{'{:02d}'.format(d.day)}.json"
            )
            # I want to make this better, more general so that anyone's results*.json can be used
            # self.prior_json.append(results_file_pattern)
        self.time_range_results_df = pd.DataFrame()
        self.null_results_list = []
        for j in self.prior_json:
            try:
                time_range_results = pd.read_json(self.results_path + j)
                self.time_range_results_df = pd.concat(
                    [self.time_range_results_df, time_range_results]
                )
            except:
                self.null_results_list.append(j)
        self.time_range_results_df = self.time_range_results_df.loc[
            self.time_range_results_df.astype(str).drop_duplicates().index
        ]
        return self.prior_json, self.time_range_results_df, self.null_results_list

    def filter_time_range_results(self):
        """INPUTS
        self.time_range_results_df --> (pd.DataFrame()) set of predictions over past lookback_days

        OUTPUTS
        todays_results --> (pd.DataFrame()) current predictions without predictions found in time_range_results
        """
        d = datetime.datetime.today()
        todays_file = f"{self.results_path}Results_{self.run_date}.json"
        # I want to make this better, more general so that anyone's results*.json can be used
        # todays_file= results_file_pattern
        if isinstance(self.todays_df, pd.DataFrame):
            todays_results = self.todays_df
        else:
            todays_results = pd.read_json(todays_file)
        if len(self.time_range_results_df):
            for e in list(self.time_range_results_df[self.static_key]):
                if e in list(todays_results[self.static_key]):
                    todays_results.drop(
                        todays_results[todays_results[self.static_key] == e].index[0],
                        inplace=True,
                        axis=0,
                    )
        else:
            print("No Prior Results")
        return todays_results


class ip_preprocessing:
    def __init__(
        self,
        tfidf_ip_count: int,
        ip_keys: list,
        ip_type: str = "",
        port_key: str = "",
        port_types: str = "",
        ip_df: pd.DataFrame = None,
    ):
        self.tfidf_ip_count = tfidf_ip_count
        self.ip_keys = ip_keys
        self.ip_type = ip_type
        self.port_key = port_key
        self.port_types = port_types

    def port_type_counter(self):
        """Simple accounting of ratio of each port type. NOT WEIGHTED BY NUMBER OF PACKETS SENT.
        INPUTS
        data --> df
        key --> df.key of port
        dicts --> mapping of port

        OUTPUTS
        data --> df with new keys key-port_types[:]
        """
        for (
            port_type
        ) in (
            self.port_types
        ):  # Generates new keys for the port type and the port classes
            port_key = f"{self.port_key}-{port_type}"
            self.data[port_key] = np.zeros((len(self.data), 1))

        def count_vals(
            rows,
        ):  # Gets the ratio of port classes for a given set of connections
            if " " in rows[self.port_key]:  # For single ports only
                port_list = rows[self.port_key].split()
            elif type(rows[self.port_key]) == str:
                port_list = [rows[self.port_key]]
            else:
                port_list = rows[self.port_key]

            count_list = np.zeros(len(self.port_types))
            for p in port_list:
                port = self.dicts[int(p)]
                for i in range(len(self.port_types)):
                    if self.port_types[i] == port:
                        count_list[i] += 1 / len(port_list)
            for i in range(len(self.port_types)):
                temp_port_key = f"{self.port_key}-{self.port_types[i]}"
                rows[temp_port_key] = count_list[i]
            rows.drop(self.port_key, inplace=True)
            return rows

        self.data = self.data.apply(count_vals, axis=1)
        return self.data

    def ip_tfidf(self):
        """Takes ip strings and preforms TFIDF on their individual quads. To generate new vocab words for octets of numbers being repeated in different quads, each quad is shifted by a factor of 256. NOTE: 0-10  were protected vocab, so each octet is also shifted by 10."""

        # Encode all ips with term frequency inverse doc freq
        @self.logger.catch(level="CRITICAL")
        def generate_sentences(row):
            offset_arr = [
                (i * 256) + 10 for i in range(4)
            ]  # separate out quads so 123.123.123 has different vals for each quad

            vals = row[f"{self.ip_type}_ip"].split(".")
            vals = [int(j) + offset_arr[i] for i, j in enumerate(vals)]
            return " ".join([str(i) for i in vals])

        @self.logger.catch(level="CRITICAL")
        def unprotect_1(
            rows,
        ):  # vectorizer has the word "1" as a protected token that is not documented, this returns no score if used
            row = rows["sentence"]
            rex = "(^1 )"
            if re.match(rex, row):
                return re.sub(
                    rex, "01 ", row
                )  # give ip 1.xxx.xxx the value of 01.xxx.xxx to avoid protected vals
            else:
                return row

        @self.logger.catch(level="CRITICAL")
        def extract(
            results: list,
            n: int,
        ):  # separate words to match words' scores
            return [item[n] for item in results]

        for k in self.ip_keys:
            self.preprocessed_df = self.preprocessed_df[
                self.preprocessed_df[k].isna() == False
            ]

        if self.tfidf_ip_count == 1:
            ip_df = self.preprocessed_df
            ip_list = list(ip_df[self.ip_keys[0]].values)

            ip_df = pd.DataFrame(ip_list, columns=[f"{self.ip_type}_ip"])
            ip_df["sentence"] = ip_df.apply(generate_sentences, axis=1)

            vocab = {f"{i}": i for i in range(1034)}

            v = vecorizer(vocabulary=vocab)
            v.fit([ip_df["sentence"][0]])
            outs = v.transform(ip_df["sentence"]).todense().tolist()
            results_list = []
            for o in outs:
                results_list.append([i for i in o if i > 0])

            quad_df = pd.DataFrame()
            for i in range(4):
                quad_df[f"quad_{i}"] = extract(results_list, i)
            self.preprocessed_df.drop(self.ip_keys[0], inplace=True, axis=1)
            self.preprocessed_df = pd.concat([self.preprocessed_df, quad_df], axis=1)
            self.logger.debug("Only 1 IP in use, TFIDF Complete.")

        if self.tfidf_ip_count == 2:
            ip_df = self.preprocessed_df

            pass
            ip_list = list(ip_df[self.ip_keys[0]].values)
            ip_list += list(
                ip_df[self.ip_keys[1]].values
            )  # use all quads as vocabulary items

            ip_df = pd.DataFrame(ip_list, columns=[f"{self.ip_type}_ip"])
            ip_df["sentence"] = ip_df.apply(generate_sentences, axis=1)

            vocab = {f"{i}": i for i in range(1034)}

            v = vecorizer(vocabulary=vocab)
            v.fit(ip_df["sentence"])
            outs = v.transform(ip_df["sentence"]).todense().tolist()
            results_list = []
            for o in outs:
                results_list.append([i for i in o if i > 0])

            quad_df = pd.DataFrame()
            for i in range(4):
                quad_df[f"quad_{i}"] = extract(results_list, i)

            dest_quad = pd.DataFrame()
            src_quad = pd.DataFrame()
            dest_quad[["dest_q0", "dest_q1", "dest_q2", "dest_q3"]] = quad_df.loc[
                : len(quad_df) // 2 - 1
            ]  # unravel total vocabulary list back to src and dest ips
            src_quad[["src_q0", "src_q1", "src_q2", "src_q3"]] = quad_df.loc[
                len(quad_df) // 2 :
            ]

            src_quad.reset_index(inplace=True)
            src_quad.drop("index", inplace=True, axis=1)
            self.dest_quad = dest_quad
            self.src_quad = src_quad

            self.preprocessed_df.drop(self.ip_keys, inplace=True, axis=1)
            self.preprocessed_df.reset_index(inplace=True)
            tfidf_keys = (
                list(self.preprocessed_df.keys().values)
                + list(self.dest_quad.keys().values)
                + list(self.src_quad.keys().values)
            )
            self.preprocessed_df = pd.concat(
                [self.preprocessed_df, dest_quad, src_quad], axis=1, ignore_index=True
            )

            self.preprocessed_df.columns = tfidf_keys
            self.preprocessed_df.drop("index", inplace=True, axis=1)
            self.logger.debug("There are 2 IPs in use, TFIDF Complete.")
        return self

    def remove_predefined_ips(self, ip_key):
        """INPUTS ip_df: pd.DataFrame(), ip_key: str OUTPUTS quads_df --> pd_DataFrame['q0','q1','q2','q3'] filtered_ips_df --> pd_DataFrame['ips'].
        Takes a dataframe and filters out private ranges, loop back ranges, link local ranges, test ranges, and multicast ranges
        as defined by: https://www.auvik.com/franklyit/blog/special-ip-address-ranges/ on 9/16/2024
        """

        # Input Data Frame Checks  all logging instances here will be at the critical level.
        def check_df_type(df, ip_keys, col_type):
            for ip_key in ip_keys:
                if isinstance(df, pd.DataFrame):
                    if df.map(type).nunique()[ip_key] > 1:
                        # loguru logger and log --> More than 1 dtype in IP Col
                        print(f"more than one type")
                        sys.exit(0)
                    else:
                        if type(df[ip_key][0]) == col_type:
                            pass
                        else:
                            # loguru logger and log --> IPS NOT STR
                            print(f"ip not str")
                            sys.exit(0)
                else:
                    # import loguru logger and log --> NOT a DF
                    print("not a pd.df")
                    sys.exit(0)

        # check_df_type(ip_df, [ip_key], str)  # check input DF

        # Input ip_key checks
        if type(self.ip_key) is str:
            if self.ip_key in self.ip_df.keys():
                pass
            else:
                # loguru log --> key not in ip_df.keys()
                logger.critical(f"{self.ip_key} not in {self.ip_df.keys()}")
                sys.exit(0)
        else:
            # loguru log --> ip_key is not a str
            logger.critical("Given key is not a str")
            sys.exit(0)

        # quads separated as integers so that logic can be applied to rm special ips
        quads_df = (
            self.ip_df[self.ip_key]
            .str.split(".", expand=True)
            .astype(int)
            .rename(columns={0: "q0", 1: "q1", 2: "q2", 3: "q3"})
        )

        # Remove special ips with /8 Private Range 10.0.0.0/8 and Loop Back Range 127.0.0.0/8
        q0_8 = [10, 127]
        for i in q0_8:
            quads_df.drop(quads_df[(quads_df["q0"] == i)].index, axis=0, inplace=True)

        # Remove special ips with /16  Private Ranges 192.168.0.0/16 and Link Local 169.254.0.0/16
        q0_16 = [192, 169]
        q1_16 = [168, 254]
        for i in range(len(q0_16)):
            quads_df.drop(
                quads_df[
                    (quads_df["q0"] == q0_16[i]) & (quads_df["q1"] == q1_16[i])
                ].index,
                axis=0,
                inplace=True,
            )

        # Remove special ips with /24  Private Ranges 192.0.0.0/24 and Test Ranges 192.0.2.0/24 198.51.100.0/24 203.0.113.0/24
        q0_24 = [192, 192, 198, 203]
        q1_24 = [0, 0, 51, 0]
        q2_24 = [0, 2, 100, 113]
        for i in range(len(q0_24)):
            quads_df.drop(
                quads_df[
                    (quads_df["q0"] == q0_24[i])
                    & (quads_df["q1"] == q1_24[i])
                    & (quads_df["q2"] == q2_24[i])
                ].index,
                axis=0,
                inplace=True,
            )

        # Remove special ips with /4 MultiCast 224.0.0.0-239.255.255.255
        q0_4 = range(224, 240)
        for i in q0_4:
            quads_df.drop(quads_df[(quads_df["q0"] == i)].index, axis=0, inplace=True)

        # Remove special ips with /12 Private Range 172.16.0.0-192.31.255.255
        q0_12 = [172]
        q1_12 = range(16, 32)
        for i in q1_12:
            quads_df.drop(
                quads_df[(quads_df["q0"] == q0_12[0]) & (quads_df["q1"] == i)].index,
                axis=0,
                inplace=True,
            )

        filtered_ips_df = pd.DataFrame()
        filtered_ips_df["ips"] = (
            quads_df["q0"].astype(str)
            + "."
            + quads_df["q1"].astype(str)
            + "."
            + quads_df["q2"].astype(str)
            + "."
            + quads_df["q3"].astype(str)
        )

        # check outputs
        if len(self.ip_df.iloc[filtered_ips_df.index]) == 0:
            print("FILTERED TO 0")
            sys.exit()
        return quads_df, filtered_ips_df.index
