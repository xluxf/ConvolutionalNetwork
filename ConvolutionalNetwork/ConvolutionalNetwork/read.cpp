#include <iostream>
#include <fstream>
#include <boost/program_options.hpp>


namespace po = boost::program_options;
using namespace std;

#define DIM 32
#define DIM_SQR 1024

int main(int argc, char* argv[])
{
	// parse commandline
	string filename;

	po::options_description desc("Allowed options");
	desc.add_options()
	("help", "produce help message")
	("file", po::value<std::string>(&filename), "set compression level");

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);    
	
	if (vm.count("help")) {
 		cout << desc << "\n";
		return 1;
	}
	
	// open file
	ifstream file (filename.c_str(), ios::in | ios::binary);
	
	if (!file.is_open())
	{
		cout << "Error opening file" << endl;
		return -1;
	}

	char label;
	char red[DIM_SQR];
	char green[DIM_SQR];
	char blue[DIM_SQR];

	// read
	while (!file.eof())
	{
		// get label and each of these channels
		file.get(label);
		file.read(red, DIM_SQR);
		file.read(green,DIM_SQR);
		file.read(blue, DIM_SQR);

		

	}

	file.close();

	return 0;
}
