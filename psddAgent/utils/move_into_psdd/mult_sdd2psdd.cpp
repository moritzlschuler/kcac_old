// To run this code after installing the PSDD C++ library run:
// g++ -no-pie paths_psdd_mult.cpp src/psdd_manager.cpp src/psdd_node.cpp src/psdd_parameter.cpp src/psdd_unique_table.cpp src/random_double_generator.cpp -Iinclude -Llib/linux -lsdd -lgmp

// change the input files approprietly

#include <iostream>
#include <unordered_map>
#include <stdio.h>
#include <dirent.h>
// #include <psdd/cnf.h>
// #include <psdd/optionparser.h>
#include <psdd/psdd_manager.h>
#include <psdd/psdd_unique_table.h>
#include <psdd/psdd_node.h>
#include <psdd/psdd_parameter.h>
#include <psdd/random_double_generator.h>
extern "C" {
#include <sdd/sddapi.h>
}
using namespace std;

int main(int argc, char **argv)
{
	// arguments must be in the form ./a.out "file.vtree" "file1.sdd" "file2.sdd" ... "fileN.sdd" "outputfile.psdd"

	// set up psdd-manager
	Vtree* vtree = sdd_vtree_read(argv[1]);
	SddManager* sdd_manager = sdd_manager_new(vtree);
	PsddManager* psdd_manager = PsddManager::GetPsddManagerFromVtree(sdd_manager_vtree(sdd_manager));

	// join the first 2 sdds to psdd
	SddNode* first_sdd = sdd_read(argv[2], sdd_manager);
	cout << "read in 1: " << argv[2] << endl;
	PsddNode* first_psdd = psdd_manager->ConvertSddToPsdd(first_sdd, sdd_manager_vtree(sdd_manager), 0);
	SddNode* current_sdd = sdd_read(argv[3], sdd_manager);
	cout << "read in 2: " << argv[3] << endl;
	PsddNode* current_psdd = psdd_manager->ConvertSddToPsdd(current_sdd, sdd_manager_vtree(sdd_manager), 0);

	pair<PsddNode*, PsddParameter> current_result_pair = psdd_manager->Multiply(current_psdd, first_psdd, 0);
	PsddParameter current_param = current_result_pair.second;
	cout << current_result_pair.first << " " << current_param.parameter() << endl;

	// define vars for next psdd
	SddNode* next_sdd;
	PsddNode* next_psdd;
	PsddParameter next_param;
	pair<PsddNode*, PsddParameter> next_result_pair;


	// conjoin remaining sdds into psdd
	for (int i = 4; i < argc - 1; i = i + 1) {
		next_sdd = sdd_read(argv[i], sdd_manager);
		cout << "read in " << i - 1 << ": " << argv[i] << endl;
		next_psdd = psdd_manager->ConvertSddToPsdd(next_sdd, sdd_manager_vtree(sdd_manager), 0);

		next_result_pair = psdd_manager->Multiply(current_result_pair.first, next_psdd, 0);	
		next_param = next_result_pair.second;
		cout << next_result_pair.first << " " << next_param.parameter() << endl;

		current_result_pair = next_result_pair;
	}

	// write out psdd to file
    psdd_node_util::WritePsddToFile(current_result_pair.first, argv[argc - 1]);

}