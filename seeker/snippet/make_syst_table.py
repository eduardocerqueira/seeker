#date: 2025-06-30T17:11:18Z
#url: https://api.github.com/gists/4c7161eab2279d5956f4bc1a3e1f6b0e
#owner: https://api.github.com/users/sanuvarghese

import os
import re
import numpy as np
from ROOT import TFile, TH1F, TMath
from collections import defaultdict

def make_syst_table_from_data_card_v1():
    basedir = "/afs/cern.ch/work/s/savarghe/private/Combine_base/CMSSW_14_1_0_pre4/src/HiggsAnalysis/CombinedLimit/Limit/FullRun2/lep/all_cards"
    shape_pattern = re.compile(r"Shapes_hcs_(\d{4})_(mu|ele)_KinFit_(Exc(?:Loose|Medium|Tight))bdt_output_WH(\d+)\.root")
    datacard_pattern = re.compile(r"datacard_hcs_(\d{4})_(mu|ele)_KinFit_bdt_output(Exc(?:Loose|Medium|Tight))_WH(\d+)\.txt")

    # Dictionary to hold matched files
    matched_files = defaultdict(lambda: {"shape": None, "datacard": None})

    # Scan directory and match files
    for filename in os.listdir(basedir):
        shape_match = shape_pattern.match(filename)
        datacard_match = datacard_pattern.match(filename)
        if shape_match:
            year, ch, ctag, mass = shape_match.groups()
            key = (int(year), ch, ctag, int(mass))
            matched_files[key]["shape"] = os.path.join(basedir, filename)
        elif datacard_match:
            year, ch, ctag, mass = datacard_match.groups()
            key = (int(year), ch, ctag, int(mass))
            matched_files[key]["datacard"] = os.path.join(basedir, filename)

    nuisances = [
        "lumi_13TeV", "lumi_16", "lumi_17", "lumi_18", "lumi_17_18", "eff", "prefire", "pujetid", "Pileup", "absmpfb",
        "absscl", "absstat", "flavorqcd", "frag", "timepteta", "pudatamc", "puptbb", "puptec1", "puptec2", "pupthf",
        "puptref", "relfsr", "relbal", "relsample", "reljerec1", "reljerec2", "relptbb", "relptec1", "relptec2", "relpthf",
        "relstatfsr", "relstathf", "relstatec", "singpiecal", "singpihcal", "JER", "norm_sig", "norm_tt ", "norm_ttg",
        "norm_tth", "norm_ttz", "norm_ttw", "norm_stop", "norm_wjet", "norm_zjet", "norm_qcd_", "norm_vv", "CP5", "hDamp",
        "topMass", "pdf", "isr", "bcstat", "bclhemuf", "bclhemur", "bcxdyb", "bcxdyc", "bcxwjc", "bcintp", "bcextp",
        "topPt", "fsr"
    ]
    corruncorr = [
        "corr", "uncorr", "uncorr", "uncorr", "corr", "corr", "corr", "corr", "corr", "corr", "corr", "uncorr", "corr",
        "corr", "uncorr", "corr", "corr", "corr", "corr", "corr", "corr", "corr", "corr", "uncorr", "uncorr", "uncorr",
        "corr", "uncorr", "uncorr", "corr", "uncorr", "uncorr", "uncorr", "corr", "corr", "uncorr", "corr", "corr", "corr",
        "corr", "corr", "corr", "corr", "corr", "corr", "corr", "corr", "corr", "corr", "corr", "corr", "corr", "uncorr",
        "corr", "corr", "corr", "corr", "corr", "uncorr", "uncorr", "corr", "corr"
    ]
    nM = 14
    ncT = 3
    syst_array = [[defaultdict(lambda: defaultdict(float)) for _ in range(nM)] for _ in range(ncT)]
    chid = [[defaultdict(int) for _ in range(nM)] for _ in range(ncT)]
    yield_data = [[[defaultdict(lambda: (0.0, 0.0)) for _ in range(nM)] for _ in range(ncT)] for _ in range(2)]
    mass = [40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 155, 160]
    category_map = {"L": "ExcLoose", "M": "ExcMedium", "T": "ExcTight"}

    for year in range(2016, 2019):
        for im in range(nM):
            for ctag in ["L", "M", "T"]:
                ctag_full = category_map[ctag]
                read_data_card(basedir, matched_files, year, "mu", mass[im], ctag_full, nuisances, corruncorr, syst_array[{"L": 0, "M": 1, "T": 2}[ctag]][im], chid[{"L": 0, "M": 1, "T": 2}[ctag]][im])
                get_yield_result(basedir, matched_files, year, "mu", mass[im], ctag_full, yield_data[0][{"L": 0, "M": 1, "T": 2}[ctag]][im], chid[{"L": 0, "M": 1, "T": 2}[ctag]][im])

        write_syst_table(year, "mu", "L", nuisances, syst_array[0], chid[0])
        write_syst_table(year, "mu", "M", nuisances, syst_array[1], chid[1])
        write_syst_table(year, "mu", "T", nuisances, syst_array[2], chid[2])

        for im in range(nM):
            for ctag in ["L", "M", "T"]:
                ctag_full = category_map[ctag]
                read_data_card(basedir, matched_files, year, "ele", mass[im], ctag_full, nuisances, corruncorr, syst_array[{"L": 0, "M": 1, "T": 2}[ctag]][im], chid[{"L": 0, "M": 1, "T": 2}[ctag]][im])
                get_yield_result(basedir, matched_files, year, "ele", mass[im], ctag_full, yield_data[1][{"L": 0, "M": 1, "T": 2}[ctag]][im], chid[{"L": 0, "M": 1, "T": 2}[ctag]][im])

        write_syst_table(year, "ele", "L", nuisances, syst_array[0], chid[0])
        write_syst_table(year, "ele", "M", nuisances, syst_array[1], chid[1])
        write_syst_table(year, "ele", "T", nuisances, syst_array[2], chid[2])

        write_yield_table(year, yield_data, chid)

    return True

def write_yield_table(year, yield_data, chid):
    def print_yield(mcname, chname, imass, out_file, yield_data):
        out_file.write(
            f"{mcname} & ${yield_data[0][0][imass][chname][0]:.1f} \\pm {yield_data[0][0][imass][chname][1]:.1f}$ "
            f"& ${yield_data[1][0][imass][chname][0]:.1f} \\pm {yield_data[1][0][imass][chname][1]:.1f}$ "
            f"& ${yield_data[0][1][imass][chname][0]:.1f} \\pm {yield_data[0][1][imass][chname][1]:.1f}$ "
            f"& ${yield_data[1][1][imass][chname][0]:.1f} \\pm {yield_data[1][1][imass][chname][1]:.1f}$ "
            f"& ${yield_data[0][2][imass][chname][0]:.1f} \\pm {yield_data[0][2][imass][chname][1]:.1f}$ "
            f"& ${yield_data[1][2][imass][chname][0]:.1f} \\pm {yield_data[1][2][imass][chname][1]:.1f}$ "
            f"\\\\\n"
        )
        return True

    os.makedirs("syst", exist_ok=True)
    with open(f"syst/mjjTable_yield_{year}.tex", "w") as out_file:
        out_file.write("\\documentclass[]{article}\n")
        out_file.write("\\usepackage{amsmath}\n")
        out_file.write("\\usepackage{array}\n")
        out_file.write("\\usepackage{multirow}\n")
        out_file.write("\\usepackage{graphicx}\n")
        out_file.write("\\usepackage{adjustbox}\n")
        out_file.write("\\usepackage{pdflscape}\n")
        out_file.write("\\usepackage[cm]{fullpage}\n")
        out_file.write("\\begin{document}\n\n\n")

        n_events = "$N_{events} \\pm stat$"
        tab1_caption = f"Event yield for exclusive charm tagging category in {year}."
        out_file.write("\\begin{table}\n")
        out_file.write(f"\\caption{{{tab1_caption}}}\n")
        out_file.write(f"\\label{{tab:sec07_eventYield_Exc_{year}}}\n")
        out_file.write("\\begin{adjustbox}{width=\\textwidth}\n")
        out_file.write("\\begin{tabular}{cc cc cc cc}\n")
        out_file.write("\\hline\n\\hline\n")
        out_file.write(f" Process & {n_events}(ExcL) & {n_events}(ExcL) & {n_events}(ExcM) & {n_events}(ExcM) & {n_events}(ExcT) & {n_events}(ExcT) \\\\\n")
        out_file.write(" & $\\mu$ + jets &  e + jets & $\\mu$ + jets &  e + jets & $\\mu$ + jets &  e + jets\\\\\n")
        out_file.write("\\hline\n")

        print_yield("$\\text{H}^{+} + \\text{H}^{-} (\\text{m}=40$ GeV)", "WH40", 0, out_file, yield_data)
        print_yield("$\\text{H}^{+} + \\text{H}^{-} (\\text{m}=50$ GeV)", "WH50", 1, out_file, yield_data)
        print_yield("$\\text{H}^{+} + \\text{H}^{-} (\\text{m}=60$ GeV)", "WH60", 2, out_file, yield_data)
        print_yield("$\\text{H}^{+} + \\text{H}^{-} (\\text{m}=70$ GeV)", "WH70", 3, out_file, yield_data)
        print_yield("$\\text{H}^{+} + \\text{H}^{-} (\\text{m}=80$ GeV)", "WH80", 4, out_file, yield_data)
        print_yield("$\\text{H}^{+} + \\text{H}^{-} (\\text{m}=90$ GeV)", "WH90", 5, out_file, yield_data)
        print_yield("$\\text{H}^{+} + \\text{H}^{-} (\\text{m}=100$ GeV)", "WH100", 6, out_file, yield_data)
        print_yield("$\\text{H}^{+} + \\text{H}^{-} (\\text{m}=110$ GeV)", "WH110", 7, out_file, yield_data)
        print_yield("$\\text{H}^{+} + \\text{H}^{-} (\\text{m}=120$ GeV)", "WH120", 8, out_file, yield_data)
        print_yield("$\\text{H}^{+} + \\text{H}^{-} (\\text{m}=130$ GeV)", "WH130", 9, out_file, yield_data)
        print_yield("$\\text{H}^{+} + \\text{H}^{-} (\\text{m}=140$ GeV)", "WH140", 10, out_file, yield_data)
        print_yield("$\\text{H}^{+} + \\text{H}^{-} (\\text{m}=150$ GeV)", "WH150", 11, out_file, yield_data)
        print_yield("$\\text{H}^{+} + \\text{H}^{-} (\\text{m}=155$ GeV)", "WH155", 12, out_file, yield_data)
        print_yield("$\\text{H}^{+} + \\text{H}^{-} (\\text{m}=160$ GeV)", "WH160", 13, out_file, yield_data)
        out_file.write("\\hline\n")
        print_yield("SM $t\\bar{t}$ + jets", "ttbar", 0, out_file, yield_data)
        print_yield("Single $t$", "stop", 0, out_file, yield_data)
        print_yield("W + jets", "wjet", 0, out_file, yield_data)
        print_yield("Z/$\\gamma$ + jets", "zjet", 0, out_file, yield_data)
        print_yield("SM $t\\bar{t}\\gamma$ + jets", "ttg", 0, out_file, yield_data)
        print_yield("SM $t\\bar{t}\\text{H}$ + jets", "tth", 0, out_file, yield_data)
        print_yield("SM $t\\bar{t}\\text{Z}$ + jets", "ttz", 0, out_file, yield_data)
        print_yield("SM $t\\bar{t}\\text{W}$ + jets", "ttw", 0, out_file, yield_data)
        print_yield("VV", "vv", 0, out_file, yield_data)
        print_yield("QCD multijet", "qcd", 0, out_file, yield_data)
        out_file.write("\\hline\n")
        out_file.write("\\end{tabular}\n")
        out_file.write("\\end{adjustbox}\n")
        out_file.write("\\end{table}\n\n\n")
        out_file.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
        out_file.write("\\end{document}\n")

    return True

def write_syst_table(year, ch, cTag, nuisances, syst_array, chid):
    def print_diff_syst(systfullname, systnickname, out_file, nuisances, syst_array, chid, ctag_idx):
        if "norm" in systnickname:
            qcd_syst = syst_array[0]["norm_qcd_"][chid[0]["qcd"]] if "qcd" in chid[0] else -1.0
            out_file.write(
                f"& {systfullname} & 6.1 & 6.1 & 6.1 & 6.1 & 6.1 & 6.1 & 6.1 & 6.1 & 6.1 & 6.1 "
                f"& 6.1 & 6.1 & 6.1 & 6.1 & 6.1 & 6.1 & 6.1 & 6.1 & 6.1 & 4.5 & 5.0 & 7.0 & 4.0 & {qcd_syst:.1f} \\\\\n"
            )
        else:
            nM = 14
            mass = [40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 155, 160]
            processes = [
                "WH40", "WH50", "WH60", "WH70", "WH80", "WH90", "WH100", "WH110", "WH120", "WH130",
                "WH140", "WH150", "WH155", "WH160", "ttbar", "ttg", "tth", "ttz", "ttw", "wjet", "zjet", "stop", "vv", "qcd"
            ]
            systs = [0.0] * 24  # 14 charged higgs + 9 MC bkg + 1 QCD
            for i, proc in enumerate(processes):
                if proc.startswith("WH"):
                    im = mass.index(int(proc[2:]))
                    systs[i] = syst_array[im][systnickname][chid[im][proc]] if proc in chid[im] else -1.0
                else:
                    # Use the chid and syst_array for the current category
                    systs[i] = syst_array[0][systnickname][chid[0][proc]] if proc in chid[0] else -1.0
            out_file.write(f"& {systfullname}")
            for isys in range(24):
                out_file.write(f" & {systs[isys]:.1f}" if systs[isys] > 0.0 else " & - ")
            out_file.write(" \\\\\n")
        return True

    channel = "muon" if "mu" in ch else "electron"
    category = {"L": "loose", "M": "medium", "T": "tight"}[cTag]
    ctag_idx = {"L": 0, "M": 1, "T": 2}[cTag]

    os.makedirs("syst", exist_ok=True)
    with open(f"syst/mjjTable_{ch}_{cTag}_{year}.tex", "w") as out_file:
        out_file.write("\\documentclass[]{article}\n")
        out_file.write("\\usepackage{amsmath}\n")
        out_file.write("\\usepackage{array}\n")
        out_file.write("\\usepackage{multirow}\n")
        out_file.write("\\usepackage{graphicx}\n")
        out_file.write("\\usepackage{adjustbox}\n")
        out_file.write("\\usepackage{pdflscape}\n")
        out_file.write("\\usepackage[cm]{fullpage}\n")
        out_file.write("\\begin{document}\n\n\n")
        tab2_caption = f"Systematic and statistical uncertainties in \\% for {channel} channel for exclusive {category} charm tagging category in {year}."
        out_file.write("\\begin{table}\n")
        out_file.write(f"\\centering\\caption{{{tab2_caption}}}\n")
        out_file.write(f"\\label{{tab:sec07_syst_Exc{cTag}_{ch}_{year}}}\n")
        out_file.write("\\begin{adjustbox}{width=\\textwidth}\n")
        out_file.write("\\begin{tabular}{ ll  cccc cccc cccc cccc cccc cccc}\n")
        out_file.write("\\hline\n")
        out_file.write(
            " Category & Nuisances "
            "& {\\rotatebox{90}{$\\text{H}^{+} + \\text{H}^{-} (\\text{m}=40$ GeV)}} & {\\rotatebox{90}{$\\text{H}^{+} + \\text{H}^{-} (\\text{m}=50$ GeV)}} "
            "& {\\rotatebox{90}{$\\text{H}^{+} + \\text{H}^{-} (\\text{m}=60$ GeV)}} & {\\rotatebox{90}{$\\text{H}^{+} + \\text{H}^{-} (\\text{m}=70$ GeV)}} "
            "& {\\rotatebox{90}{$\\text{H}^{+} + \\text{H}^{-} (\\text{m}=80$ GeV)}} & {\\rotatebox{90}{$\\text{H}^{+} + \\text{H}^{-} (\\text{m}=90$ GeV)}} "
            "& {\\rotatebox{90}{$\\text{H}^{+} + \\text{H}^{-} (\\text{m}=100$ GeV)}} & {\\rotatebox{90}{$\\text{H}^{+} + \\text{H}^{-} (\\text{m}=110$ GeV)}} "
            "& {\\rotatebox{90}{$\\text{H}^{+} + \\text{H}^{-} (\\text{m}=120$ GeV)}} & {\\rotatebox{90}{$\\text{H}^{+} + \\text{H}^{-} (\\text{m}=130$ GeV)}} "
            "& {\\rotatebox{90}{$\\text{H}^{+} + \\text{H}^{-} (\\text{m}=140$ GeV)}} & {\\rotatebox{90}{$\\text{H}^{+} + \\text{H}^{-} (\\text{m}=150$ GeV)}} "
            "& {\\rotatebox{90}{$\\text{H}^{+} + \\text{H}^{-} (\\text{m}=155$ GeV)}} & {\\rotatebox{90}{$\\text{H}^{+} + \\text{H}^{-} (\\text{m}=160$ GeV)}} "
            "& {\\rotatebox{90}{SM $t\\bar{t}$ + jets}} "
            "& {\\rotatebox{90}{SM $t\\bar{t}\\gamma$ + jets}} "
            "& {\\rotatebox{90}{SM $t\\bar{t}\\text{H}$ + jets}} "
            "& {\\rotatebox{90}{SM $t\\bar{t}\\text{Z}$ + jets}} "
            "& {\\rotatebox{90}{SM $t\\bar{t}\\text{W}$ + jets}} "
            "& {\\rotatebox{90}{W + jets}} "
            "& {\\rotatebox{90}{Z/$\\gamma$ + jets}} "
            "& {\\rotatebox{90}{Single $t$}} "
            "& {\\rotatebox{90}{VV}} "
            "& {\\rotatebox{90}{QCD multijet}} "
            "\\\\ \n"
        )
        out_file.write("\\hline\n")
        out_file.write(f"\\multicolumn{{26}}{{c}}{{{channel} + jets }} \\\\\n")
        out_file.write("\\hline\n")
        out_file.write(f"{category}\n")

        print_diff_syst("Luminosity (Run2)", "lumi_13TeV", out_file, nuisances, syst_array, chid, ctag_idx)
        if year == 2016:
            print_diff_syst("Luminosity (2016)", "lumi_16", out_file, nuisances, syst_array, chid, ctag_idx)
        if year == 2017:
            print_diff_syst("Luminosity (2017)", "lumi_17", out_file, nuisances, syst_array, chid, ctag_idx)
            print_diff_syst("Luminosity (2017-18)", "lumi_17_18", out_file, nuisances, syst_array, chid, ctag_idx)
        if year == 2018:
            print_diff_syst("Luminosity (2018)", "lumi_18", out_file, nuisances, syst_array, chid, ctag_idx)
            print_diff_syst("Luminosity (2017-18)", "lumi_17_18", out_file, nuisances, syst_array, chid, ctag_idx)
        if year != 2018:
            print_diff_syst("Prefire", "prefire", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("Pileup", "Pileup", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("Lepton", "eff", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("Pileup jet identification", "pujetid", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("JES:Absolute MPF Bias", "absmpfb", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("JES:Absolute Scale", "absscl", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("JES:Absolute Statistics", "absstat", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("JES:Flavor QCD", "flavorqcd", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("JES:Fragmentation", "frag", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("JES:TimePtEta", "timepteta", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("JES:Pileup Data/MC", "pudatamc", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("JES:Pileup Pt bb", "puptbb", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("JES:Pileup Pt EC1", "puptec1", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("JES:Pileup Pt EC2", "puptec2", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("JES:Pileup Pt HF", "pupthf", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("JES:Pileup Pt ref", "puptref", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("JES:Relative FSR", "relfsr", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("JES:Relative Bal", "relbal", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("JES:Relative Sample", "relsample", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("JES:Relative JER EC1", "reljerec1", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("JES:Relative JER EC2", "reljerec2", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("JES:Relative Pt bb", "relptbb", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("JES:Relative Pt EC1", "relptec1", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("JES:Relative Pt EC2", "relptec2", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("JES:Relative Pt HF", "relpthf", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("JES:Relative Stat FSR", "relstatfsr", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("JES:Relative Stat HF", "relstathf", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("JES:Relative Stat EC", "relstatec", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("JES:Single pion ECAL", "singpiecal", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("JES:Single pion HCAL", "singpihcal", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("JER", "JER", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("$bc$ tagging stat.", "bcstat", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("$bc$ tagging XS DY-b", "bcxdyb", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("$bc$ tagging XS DY-c", "bcxdyc", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("$bc$ tagging XS Wjet-c", "bcxwjc", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("$bc$ tagging interpolation", "bcintp", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("$bc$ tagging extrapolation", "bcextp", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("Renomalization", "bclhemur", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("Factorization", "bclhemuf", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("PDF", "pdf", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("ISR", "isr", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("FSR", "fsr", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("CP5", "CP5", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("$h_{\\text{damp}}$", "hDamp", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("top quark mass", "topMass", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("top $p_T$ reweight", "topPt", out_file, nuisances, syst_array, chid, ctag_idx)
        print_diff_syst("Normalization", "norm", out_file, nuisances, syst_array, chid, ctag_idx)
        out_file.write("\\hline\n\\hline\n")
        out_file.write("\\end{tabular}\n")
        out_file.write("\\end{adjustbox}\n")
        out_file.write("\\end{table}\n\n\n")
        out_file.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
        out_file.write("\\end{document}\n")

    return True

def get_yield_result(basedir, matched_files, year, ch, mass, ctag, yield_data, chid):
    key = (year, ch, ctag, mass)
    infile = matched_files[key]["shape"]
    if not infile or not os.path.exists(infile):
        print(f"Error: Shape file for {key} not found or inaccessible")
        return False

    print(f"Infile: {infile}")
    fin = TFile.Open(infile)
    if not fin or fin.IsZombie():
        print(f"Error: Cannot open shape file {infile}")
        return False

    for ch_name, _ in chid.items():
        h1 = fin.Get(ch_name)
        if not h1:
            print(f"Warning: Histogram {ch_name} not found in {infile}")
            yield_data[ch_name] = (0.0, 0.0)
            continue
        error = np.array([0.0], dtype=float)
        integral = h1.IntegralAndError(1, h1.GetNbinsX(), error)
        yield_data[ch_name] = (integral, error[0])
    fin.Close()

    return True

def read_data_card(basedir, matched_files, year, ch, mass, ctag, nuisances, corruncorr, syst_array, chid):
    key = (year, ch, ctag, mass)
    infile = matched_files[key]["datacard"]
    infileshape = matched_files[key]["shape"]

    if not infile or not os.path.exists(infile):
        print(f"Error: Datacard file for {key} not found or inaccessible")
        return False
    if not infileshape or not os.path.exists(infileshape):
        print(f"Error: Shape file for {key} not found or inaccessible")
        return False

    finshape = TFile.Open(infileshape)
    if not finshape or finshape.IsZombie():
        print(f"Error: Cannot open shape file {infileshape}")
        return False

    # Define nuisances requiring year and category suffixes
    bc_nuisances = ["bcstat", "bcxdyb", "bcxdyc", "bcxwjc", "bcintp", "bcextp", "bclhemuf", "bclhemur"]
    year_nuisances = ["relstatec", "relsample", "relptec1", "relptec2", "absstat", "timepteta", "relstatfsr", "relstathf" , "reljerec1" , "reljerec2"]
    jer_nuisance = ["JER"]

    syst_p_name = [None] * 11
    with open(infile, "r") as fin:
        print(f"Reading datacard: {infile}")
        for line in fin:
            if "process" in line and "ttbar" in line:
                parts = line.strip().split()
                syst_p_name = parts[1:12]
                print(f"Processes found in datacard: {syst_p_name}")
                for iv, name in enumerate(syst_p_name):
                    chid[name] = iv

            if "lnN" in line:
                for isys, nuisance in enumerate(nuisances):
                    systname = f"CMS_{nuisance}"
                    if systname in line:
                        parts = line.strip().split()
                        syst_p_s = parts[2:13]
                        ch_syst = {}
                        print(f"lnN nuisance {systname}: {syst_p_s}")
                        for iv in range(11):
                            if syst_p_s[iv] != "-":
                                syst_p = float(syst_p_s[iv])
                                syst_percent = (syst_p - 1.0) * 100.0
                                ch_syst[iv] = syst_percent
                            else:
                                ch_syst[iv] = -1.0
                        syst_array[nuisance] = ch_syst

            if "shape" in line and "shapes" not in line:
                for isys, (nuisance, cuc) in enumerate(zip(nuisances, corruncorr)):
                    systname = nuisance
                    is_uncorr = "uncorr" in cuc
                    if systname in line:
                        parts = line.strip().split()
                        syst_p_s = parts[2:13]
                        ch_syst = {}
                        print(f"Shape nuisance {systname}: {syst_p_s}")
                        for iv in range(11):
                            hNNom = syst_p_name[iv]
                            if hNNom is None:
                                print(f"Warning: Process index {iv} not defined in datacard")
                                ch_syst[iv] = -1.0
                                continue
                            if syst_p_s[iv] != "-":
                                # Adjust histogram name based on nuisance type
                                if nuisance in bc_nuisances:
                                    hNUp = f"{hNNom}_{nuisance}_{year}_{ctag}Up"
                                    hNDown = f"{hNNom}_{nuisance}_{year}_{ctag}Down"
                                elif nuisance in year_nuisances:
                                    hNUp = f"{hNNom}_{nuisance}_{year}Up"
                                    hNDown = f"{hNNom}_{nuisance}_{year}Down"
                                elif nuisance in jer_nuisance:
                                    hNUp = f"{hNNom}_{nuisance}_{year}Up"
                                    hNDown = f"{hNNom}_{nuisance}_{year}Down"
                                else:
                                    hNUp = f"{hNNom}_{systname}Up"
                                    hNDown = f"{hNNom}_{systname}Down"

                                hNom = finshape.Get(hNNom)
                                hUp = finshape.Get(hNUp)
                                hDown = finshape.Get(hNDown)

                                if not hNom:
                                    print(f"========== base: {hNNom} hist not found in {infileshape} =========")
                                    ch_syst[iv] = -1.0
                                    continue
                                if not hUp:
                                    print(f"========== up: {hNUp} hist not found in {infileshape} =========")
                                    ch_syst[iv] = -1.0
                                    continue
                                if not hDown:
                                    print(f"========== down: {hNDown} hist not found in {infileshape} =========")
                                    ch_syst[iv] = -1.0
                                    continue

                                maxUp = 0.0
                                maxDown = 0.0
                                for ibin in range(1, hNom.GetNbinsX() + 1):
                                    up_bin = abs(hUp.GetBinContent(ibin)) if hUp else 0.0
                                    down_bin = abs(hDown.GetBinContent(ibin)) if hDown else 0.0
                                    base_bin = abs(hNom.GetBinContent(ibin))
                                    ratio_up = min(base_bin / up_bin, up_bin / base_bin) if up_bin > 0 else 0
                                    ratio_down = min(base_bin / down_bin, down_bin / base_bin) if down_bin > 0 else 0
                                    frac_up = 1.0 - ratio_up
                                    frac_down = 1.0 - ratio_down
                                    maxUp = max(frac_up, maxUp)
                                    maxDown = max(frac_down, maxDown)

                                errorP = 100.0 * max(maxUp, maxDown)
                                ch_syst[iv] = errorP
                            else:
                                ch_syst[iv] = -1.0
                        syst_array[nuisance] = ch_syst

    finshape.Close()
    return True

if __name__ == "__main__":
    make_syst_table_from_data_card_v1()