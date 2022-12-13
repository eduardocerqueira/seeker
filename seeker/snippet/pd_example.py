#date: 2022-12-13T16:56:01Z
#url: https://api.github.com/gists/5d43f41f04cb5621030c1a1065285b90
#owner: https://api.github.com/users/DannyVanpoucke

"""
This is a basic example of how to create, plot, and analyze Phase Diagrams using the pymatgen
codebase and Materials Project database. To run this example, you should:

* have pymatgen (www.pymatgen.org) installed along with matplotlib
* obtain a Materials Project API key (https://www.materialsproject.org/open)
* paste that API key in the MAPI_KEY variable below, e.g. MAPI_KEY = "foobar1234"

For citation, see https://www.materialsproject.org/citing
For the accompanying comic book, see http://www.hackingmaterials.com/pdcomic
"""

from pymatgen import MPRester
from pymatgen.entries.compatibility import MaterialsProjectCompatibility
from pymatgen.phasediagram.pdanalyzer import PDAnalyzer
from pymatgen.phasediagram.pdmaker import PhaseDiagram
from pymatgen.phasediagram.plotter import PDPlotter

if __name__ == "__main__":
    MAPI_KEY = None  # You must change this to your Materials API key! (or set MAPI_KEY env variable)
    system = ["Fe", "P"]  # system we want to get PD for
    # system = ["Fe", "P", "O"]  # alternate system example: ternary

    mpr = MPRester(MAPI_KEY)  # object for connecting to MP Rest interface
    compat = MaterialsProjectCompatibility()  # sets energy corrections and +U/pseudopotential choice

    # Create phase diagram!
    unprocessed_entries = mpr.get_entries_in_chemsys(system)
    processed_entries = compat.process_entries(unprocessed_entries)  # filter and add energy corrections
    pd = PhaseDiagram(processed_entries)

    # Plot!
    plotter = PDPlotter(pd, show_unstable=False)  # you can also try show_unstable=True
    plotter.show()
    # plotter.write_image("{}.png".format('-'.join(system)), "png")  # save figure

    # Analyze phase diagram!
    pda = PDAnalyzer(pd)

    print 'Stable Entries (formula, materials_id)\n--------'
    for e in pd.stable_entries:
        print e.composition.reduced_formula, e.entry_id

    print '\nUnstable Entries (formula, materials_id, e_above_hull (eV/atom), decomposes_to)\n--------'
    for e in pd.unstable_entries:
        decomp, e_above_hull = pda.get_decomp_and_e_above_hull(e)
        pretty_decomp = [("{}:{}".format(k.composition.reduced_formula, k.entry_id), round(v, 2)) for k, v in decomp.iteritems()]
        print e.composition.reduced_formula, e.entry_id, "%.3f" % e_above_hull, pretty_decomp