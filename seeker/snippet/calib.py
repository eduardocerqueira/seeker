#date: 2021-10-21T17:01:30Z
#url: https://api.github.com/gists/69871fcbef87b1dee5055923c39c141c
#owner: https://api.github.com/users/riga

# coding: utf-8

from array import array

from scinum import Number
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch()


# preparations #####################################################################################


shapes_file = "shapes_20.root"
shapes_file_unweighted = "shapes_unweighted_20.root"

cat_name = "hh_ggf_2018_mutau_category"
signal = "ggHH_kl_1_kt_1_hbbhtt"
backgrounds = ["tt_fh", "tt_sl", "tt_dl", "dy", "tth_bb"]


def make_list(obj):
    return list(obj) if isinstance(obj, (list, tuple, set)) else [obj]


def get_background_hist(f, cat_name):
    cat_dir = f.Get(cat_name)
    assert bool(cat_dir)

    b_hist = None
    for name in backgrounds:
        h = cat_dir.Get(name)
        if not h:
            continue
        if not b_hist:
            b_hist = h.Clone()
        else:
            b_hist.Add(h)
    assert bool(b_hist)

    return b_hist


def get_signal_hist(f, cat_name):
    cat_dir = f.Get(cat_name)
    assert bool(cat_dir)

    s_hist = cat_dir.Get(signal)
    assert bool(s_hist)

    return s_hist


def normalize_hist(h):
    h.Scale(1. / h.Integral())


def plot_calibration_curve(path, points, inner_texts=None, top_right_text=None):
    import plotlib.root as r

    # start plotting
    r.setup_style()
    canvas, (pad,) = r.routines.create_canvas()
    pad.cd()
    draw_objs = []

    # dummy histogram to control axes
    h_dummy = ROOT.TH1F("dummy", ";Discriminator;S / (S + B)", 1, 0., 1.)
    r.setup_hist(h_dummy, pad=pad, props={"LineWidth": 0, "Minimum": 0., "Maximum": 1.})
    draw_objs.append((h_dummy, "HIST"))

    # diagonal line indicating perfect calibration
    c_line = ROOT.TLine(0., 0., 1., 1.)
    r.setup_line(c_line, props={"NDC": False, "LineColor": 16})
    draw_objs.append((c_line, "L"))

    # convert the points to a TGraph
    graph = ROOT.TGraphErrors(
        len(points),
        array("f", [p[0] for p in points]),  # x values
        array("f", [p[1] for p in points]),  # y values
        array("f", [0.] * len(points)),  # x errors
        array("f", [p[2] for p in points]),  # y errors
    )
    r.setup_graph(graph, props={"MarkerStyle": 20})
    draw_objs.append((graph, "PLEZ"))

    # inner labels
    if inner_texts:
        for i, text in enumerate(make_list(inner_texts)):
            inner_label = r.routines.create_top_left_label(text, pad=pad, x_offset=25,
                y_offset=40 + i * 26, props={"TextSize": 18})
            draw_objs.append(inner_label)

    # top right label
    if top_right_text:
        top_right_label = r.routines.create_top_right_label(top_right_text, pad=pad)
        draw_objs.append(top_right_label)

    # cms label
    cms_labels = r.routines.create_cms_labels(layout="outside_horizontal", postfix="Private work",
        pad=pad)
    draw_objs.extend(cms_labels)

    # draw all objects
    r.routines.draw_objects(draw_objs)

    # save
    r.update_canvas(canvas)
    canvas.SaveAs(path)


# actual algorithm #################################################################################


# storage for points as a list of (c, r, err_r) tuples
points = []

# open files
f = ROOT.TFile(shapes_file, "READ")
f_unweighted = ROOT.TFile(shapes_file_unweighted, "READ")

# get the actual signal and background hists
s_hist = get_signal_hist(f, cat_name)
b_hist = get_background_hist(f, cat_name)
assert s_hist.GetNbinsX() == b_hist.GetNbinsX()

# get the unweighted signal and background hists
s_hist_unweighted = get_signal_hist(f_unweighted, cat_name)
b_hist_unweighted = get_background_hist(f_unweighted, cat_name)
assert s_hist_unweighted.GetNbinsX() == b_hist_unweighted.GetNbinsX()

# normalize them
normalize_hist(s_hist)
normalize_hist(b_hist)

# get r values
n_points = s_hist.GetNbinsX()
for i in range(1, n_points + 1):
    # get values
    c = s_hist.GetBinCenter(i)
    s = s_hist.GetBinContent(i)
    b = b_hist.GetBinContent(i)
    r = s / (s + b)

    # estimate the error
    ns = s_hist_unweighted.GetBinContent(i)
    nb = b_hist_unweighted.GetBinContent(i)
    ns = Number(ns, {"s": ns**0.5})
    nb = Number(nb, {"b": nb**0.5})
    r_unweighted = ns / (ns + nb)
    err_r_rel = r_unweighted(direction="up", diff=True, factor=True)

    # add the point
    points.append((c, r, r * err_r_rel))

# plot
plot_calibration_curve("calib.png", points, inner_texts=[
    "Category: {}".format(cat_name),
    "Signal: {}".format(signal),
])
print(points)
