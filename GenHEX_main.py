from GenHEX_wrapper_functions import *
from setup_UI import *
from estimate_GGPs import *
from evaluate_HEX import *

def main(function='Run', filename='default_setup.xlsx', setup=None):
    if function == 'Create setup file':
        create_setup_excel(filename=filename)
    elif function == 'Run UI':
        UI_main()
    elif function == 'Estimate GGPs':
        setup = load_setup_from_excel(filename)
        setup = estimate_GGPS(setup)
        save_output_objects_to_excel(filename, setup)
    elif function == 'Run':
        setup = load_setup_from_excel(filename)
        setup = evaluate_HEX(setup)
        save_output_objects_to_excel(filename, setup)
    else:
        print("Not a valid function")


