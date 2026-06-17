# genHEX
Methods and data for generalized design of heat exchangers. 

Related publications:  
_Generalized Method for the Conceptual Design of Compact Heat Exchangers_  
Petter Miltén, Isak Johnsson, Anders Lundbladh, Carlos Xisto  
J. Eng. Gas Turbines Power. Nov 2024, 146(11) 

_Conceptual Design Exploration of Hydrogen Enhanced Intercooling for Future Aeroengines_  
Petter Miltén, Isak Johnsson, Anders Lundbladh, Carlos Xisto  
J. Eng. Gas Turbines Power. Nov 2025, 147(12)  

_Conceptual design of the hydrogen-enhanced intercooler_  
Petter Miltén 
Thesis for the degree of licentiate 

_Modelling hydrogen fuel cell aircraft in suave_  
Christian Svensson, Petter Miltén, Tomas Grönstedt 
Proceedings of the 34th Congress of the International Council of the Aeronautical Sciences

_GenHEX: A new heat exchanger design framework_  
Petter Miltén, Isak Johnsson, Anders Lundbladh, Carlos Xisto  
Swedish Aerospace Conference, FT2025 215


Repository is under constant development. If you have any questions or remarks, please email me at milten@chalmers.se

------- HOW TO USE IT -------
The easiest setup method is:

from GenHEX_main import *

//if you want to create a new setup file, just edit the filename and run
main(function="Create setup file", filename="Default.xlsx")
//you could also copy and rename another setup file
//after changing the fluid and hex inputs, CLOSE the xlsx file and run

// Run in deign mode:
// Will estimate what Generalized Geometrical Parameters is suitable for your application
main(function="Estimate GGPs", filename="Default.xlsx")

// Run in off-deign mode:
// Requires a defined heat exchanger, but keeps it fixed and allows for varying the fluid stream properties
main(function="Run", filename="Default.xlsx")

//all output parameters are written to the setup file but in output tabs
