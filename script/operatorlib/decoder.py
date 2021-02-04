

class SoftInformation:
    def __init__(self, genomes):
        self.num_cp = decode_cp(genomes[0:2])
        self.num_pop = decode_n_p(genomes[2:4])
        self.num_individual = decode_n_i(genomes[4:7])
        self.method_divide_pop = decode_di(genomes[7:9])
        self.curve_type = decode_curve_type(genomes[9:11])
        self.so = decode_so(genomes[11:13])
        self.so_param = decode_so_param(genomes[13:16], self.so)
        self.elitism = decode_elitism(genomes[16:18])
        self.rank = decode_rank(genomes[18:20])
        self.se, self.se_param = decode_se(genomes[20:22], genomes[22:24])
        self.exploit = decode_exploit(genomes[24:27])
        self.exploit_param = decode_exploit_param(genomes[27:29], self.exploit)
        self.twins = decode_twins(genomes[29:30])
        self.explore, self.explore_param = decode_explore(genomes[30:33], genomes[33:35])
        self.infer = decode_infer(genomes[35:36])
        self.end, self.end_param = decode_end(genomes[36:37], genomes[37:40])
        self.case1_premature, self.case1_premature_param = \
            decode_premature(genomes[40:42], genomes[42:44])
        if self.num_pop == 1:
            self.case2_similar, self.case2_similar_param = None, None
        else:
            self.case2_similar, self.case2_similar_param = \
                decode_similar(genomes[44:46], genomes[46:48])
        self.case3_restart = decode_restart(genomes[48:49])
        self.cell = decode_cellular(genomes[49:50])
        self.inject = decode_injection(genomes[50:52])
        self.repair = decode_repair(genomes[52:54])
        self.migration = decode_migrate(genomes[54:56])
        self.anti = decode_anti(genomes[56:58])
        self.fbcl = decode_fbcl(genomes[58:59])
        self.decy = decode_decy(genomes[59:61])


def decode_cp(_cp):
    dict_cp = {'00': 'per_10',
               '01': 'per_5',
               '10': 'per_2',
               '11': 'per_1'}
    return dict_cp[_cp]


def decode_n_p(_n_p):
    dict_n_p = {'00': 1,
                '01': 2,
                '10': 5,
                '11': 10}
    return dict_n_p[_n_p]


def decode_n_i(_n_i):
    dict_n_i = {'000': 5,
                '001': 10,
                '010': 20,
                '011': 40,
                '100': 100,
                '101': 200,
                '110': 400,
                '111': 600}
    return dict_n_i[_n_i]


def decode_di(_di):
    dict_di = {'00': 'default',
               '01': "island_model",
               '10': "maps",
               '11': "cegda"}
    return dict_di[_di]


def decode_curve_type(_curve_type):
    dict_c_t = {'00': 'six_bezier',
                '01': 'b_spline',
                '10': 'rts',
                '11': 'tangent_circle_curve'}
    return dict_c_t[_curve_type]


def decode_so(_so):
    dict_so = {'00': 'penalty_sort',
               '01': 'non_dominated_sort',
               '10': 'alpha_level_sort',
               '11': 'num_of_un_constraint'}
    return dict_so[_so]


def decode_so_param(_so_param, so):
    if so == 'penalty_sort':
        dict_so_param = {'000': "rk_linear_1000",
                         '001': "rk_linear_5000",
                         '010': "rk_sigmoid_1000",
                         '011': "rk_sigmoid_5000",
                         '100': "rk_ex2_1000",
                         '101': "rk_ex2_5000",
                         '110': "rk_ex3_1000",
                         '111': "rk_ex3_5000"}
    elif so == "no_dominated_sort":
        dict_so_param = {'000': "ns_1",
                         '001': "ns_2",
                         '010': "ns_3",
                         '011': "ns_4",
                         '100': "ns_5",
                         '101': "ns_6",
                         '110': "ns_7",
                         '111': "ns_8"}
    elif so == "alpha_level_sort":
        dict_so_param = {'000': "compute_alpha_1",
                         '001': "compute_alpha_2",
                         '010': "compute_alpha_3",
                         '011': "compute_alpha_4",
                         '100': "compute_alpha_5",
                         '101': "compute_alpha_6",
                         '110': "compute_alpha_7",
                         '111': "compute_alpha_8"}
    else:
        dict_so_param = {'000': "nuc_1",
                         '001': "nuc_2",
                         '010': "nuc_3",
                         '011': "nuc_4",
                         '100': "nuc_5",
                         '101': "nuc_6",
                         '110': "nuc_7",
                         '111': "nuc_8"}
    return dict_so_param[_so_param]


def decode_elitism(_elitism):
    dict_elitism = {'00': 0,
                    '01': 0.001,
                    '10': 0.1,
                    '11': 0.2}
    return dict_elitism[_elitism]


def decode_rank(_rank):
    dict_rank = {'00': "linear_rank",
                 '01': "sqrt_rank",
                 '10': "ex_rank",
                 '11': "reciprocal_rank"}
    return dict_rank[_rank]


def decode_se(_se, _se_param):
    dict_se = {'00': "truncation_selection",
               '01': "tournament_selection",
               '10': "roulette_wheel_selection",
               '11': "stochastic_universal_selection"}
    if _se == "00":
        dict_se_param = {'00': 0.9,
                         '01': 0.7,
                         '10': 0.5,
                         '11': 0.3}
    elif _se == "01":
        dict_se_param = {'00': 1,
                         '01': 2,
                         '10': 3,
                         '11': 4}
    elif _se == "10":
        dict_se_param = {'00': 'fitness_no_duplication',
                         '01': 'fitness_allow_duplication',
                         '10': 'ranking_system_no_duplication',
                         '11': 'ranking_system_allow_duplication'}
    else:
        dict_se_param = {'00': 2,
                         '01': 3,
                         '10': 4,
                         '11': 5}
    return dict_se[_se], dict_se_param[_se_param]


def decode_se_param(_se_param):
    dict_se = {'00': "truncation_selection",
               '01': "tournament_selection",
               '10': "roulette_wheel_selection",
               '11': "stochastic_universal_selection"}
    return dict_se[_se_param]


def decode_exploit(_exploit):
    dict_exploit = {'000': 'npx',
                    '001': "ux",
                    '010': "ax",
                    '011': "pso_exploit",
                    '100': "safari",
                    '101': "commensalism",
                    '110': "de_rand",
                    '111': "de_best"}
    return dict_exploit[_exploit]


def decode_exploit_param(_exploit_param, exploit):
    if exploit == 'npx':
        dict_exploit_param = {'00': 0,
                              '01': 1,
                              '10': 2,
                              '11': 3}
    elif exploit == 'ux':
        dict_exploit_param = {'00': 0.5,
                              '01': 0.6,
                              '10': 0.7,
                              '11': 0.8}
    elif exploit == 'ax':
        # TODO
        dict_exploit_param = {'00': 0.5,
                              '01': 0.6,
                              '10': 0.7,
                              '11': 0.8}
    elif exploit == 'pso_exploit':
        # TODO
        dict_exploit_param = {'00': 0.5,
                              '01': 0.6,
                              '10': 0.7,
                              '11': 0.8}
    elif exploit == 'safari':
        # TODO
        dict_exploit_param = {'00': 0.5,
                              '01': 0.6,
                              '10': 0.7,
                              '11': 0.8}
    elif exploit == 'commensalism':
        # TODO
        dict_exploit_param = {'00': 0.5,
                              '01': 0.6,
                              '10': 0.7,
                              '11': 0.8}
    elif exploit == 'de_rand':
        # TODO
        dict_exploit_param = {'00': 0.5,
                              '01': 0.6,
                              '10': 0.7,
                              '11': 0.8}
    else:
        # TODO
        dict_exploit_param = {'00': 0.5,
                              '01': 0.6,
                              '10': 0.7,
                              '11': 0.8}
    return dict_exploit_param[_exploit_param]


def decode_twins(_twins):
    return int(_twins)


def decode_explore(_explore, _explore_param):
    dict_explore = {'000': "num",
                    '001': "um",
                    '010': "gm",
                    '011': "cm",
                    '100': "pus",
                    '101': "sgwo",
                    '110': "cinf",
                    '111': "none_explore"}
    if _explore == '000' or _explore == '001':
        dict_explore_param = {'00': "fixed_pm_1",
                              '01': "fixed_pm_2",
                              '10': "adaptive_pm_1",
                              '11': "adaptive_pm_2"}
    elif _explore == '010':
        dict_explore_param = {'00': "fixed_pm_1",
                              '01': "fixed_pm_2",
                              '10': "adaptive_pm_1",
                              '11': "adaptive_pm_2"}
    elif _explore == '011':
        # TODO
        dict_explore_param = {'00': "fixed_pm_1",
                              '01': "fixed_pm_2",
                              '10': "adaptive_pm_1",
                              '11': "adaptive_pm_2"}
    elif _explore == '100':
        dict_explore_param = {'00': "fixed_pm_1",
                              '01': "fixed_pm_2",
                              '10': "adaptive_pm_1",
                              '11': "adaptive_pm_2"}
    elif _explore == '101':
        dict_explore_param = {'00': "fixed_pm_1",
                              '01': "fixed_pm_2",
                              '10': "adaptive_pm_1",
                              '11': "adaptive_pm_2"}
    elif _explore == '110':
        dict_explore_param = {'00': "fixed_pm_1",
                              '01': "fixed_pm_2",
                              '10': "adaptive_pm_1",
                              '11': "adaptive_pm_2"}
    else:
        dict_explore_param = {'00': "fixed_pm_1",
                              '01': "fixed_pm_2",
                              '10': "adaptive_pm_1",
                              '11': "adaptive_pm_2"}
    return dict_explore[_explore], dict_explore_param[_explore_param]


def decode_infer(_infer):
    return int(_infer)


def decode_end(_end, _end_param):
    dict_end = {"0": "gen",
                "1": "time"}
    dict_end_param = {'000': 20,
                      '001': 40,
                      '010': 60,
                      '011': 80,
                      '100': 100,
                      '101': 150,
                      '110': 200,
                      '111': 500}
    return dict_end[_end], dict_end_param[_end_param]


def decode_premature(_case1, _case1_param):
    dict_case1 = {'00': "do_nothing",
                  '01': "evolution_stagnation",
                  '10': "population_homogenization",
                  '11': "goal_achievement"}
    if _case1 == '00':
        dict_case1_param = {'00': 0,
                            '01': 0,
                            '10': 0,
                            '11': 0}
    elif _case1 == '01':
        dict_case1_param = {'00': 3,
                            '01': 5,
                            '10': 10,
                            '11': 20}
    elif _case1 == "10":
        dict_case1_param = {'00': 0.5,
                            '01': 0.6,
                            '10': 0.8,
                            '11': 0.9}
    else:
        dict_case1_param = {'00': 1.3,
                            '01': 1.2,
                            '10': 1.15,
                            '11': 1.1}
    return dict_case1[_case1], dict_case1_param[_case1_param]


def decode_similar(_case2, _case2_param):
    dict_case1 = {'00': "default_case2",
                  '01': "reset",
                  '10': "killing",
                  '11': "adjust"}
    dict_case1_param = {'00': 0.80,
                        '01': 0.85,
                        '10': 0.90,
                        '11': 0.95}
    return dict_case1[_case2], dict_case1_param[_case2_param]


def decode_restart(_restart):
    return int(_restart)


def decode_cellular(_cell):
    return int(_cell)


def decode_injection(_inject):
    dict_inject = {'00': 0,
                   '01': 0.1,
                   '10': 0.2,
                   '11': 0.3}
    return dict_inject[_inject]


def decode_repair(_repair):
    dict_repair = {'00': "do_nothing_in_repair",
                   '01': "greedy_search",
                   '10': "random_search",
                   '11': "heuristic_search"}
    return dict_repair[_repair]


def decode_migrate(_migrate):
    dict_migrate = {'00': 0,
                    '01': 0.00001,
                    '10': 0.1,
                    '11': 0.3}
    return dict_migrate[_migrate]


def decode_anti(_anti):
    dict_anti = {'00': 0,
                 '01': 0.00001,
                 '10': 0.1,
                 '11': 0.3}
    return dict_anti[_anti]


def decode_fbcl(_fbcl):
    return int(_fbcl)


def decode_decy(_decy):
    dict_decy = {'00': 1,
                 '01': 0.99,
                 '10': 0.95,
                 '11': 0.9}
    return dict_decy[_decy]


def decode_pfih(_pfih):
    dict_pfih = {'000': "pfih_1",
                 '001': "pfih_2",
                 '010': "pfih_3",
                 '011': "pfih_4",
                 '100': "pfih_5",
                 '101': "pfih_6",
                 '110': "pfih_7",
                 '111': "pfih_8"}
    return dict_pfih[_pfih]


if __name__ == "__main__":
    genome_64_bits = '1000000000' \
                     '0000101000' \
                     '0000000100' \
                     '0000000010' \
                     '0000000000' \
                     '0000000000' \
                     '0000'
    soft = SoftInformation(genome_64_bits)
