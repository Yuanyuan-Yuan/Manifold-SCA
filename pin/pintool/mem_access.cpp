//
// This tool counts the number of times a routine is executed and 
// the number of instructions executed in a routine
//
#include <fstream>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <string.h>
#include "pin.H"
using std::ofstream;
using std::string;
using std::hex;
using std::ios;
using std::setw;
using std::cerr;
using std::dec;
using std::endl;
ofstream outFile;

// string TARGET_IMG = "ffmpeg_g";
// string TARGET_RTN = "";

static std::unordered_map<ADDRINT, std::string> str_of_ins_at;
static std::unordered_map<ADDRINT, std::string> str_of_func_at;
    
const char * StripPath(const char * path)
{
    const char * file = strrchr(path,'/');
    if (file)
        return file+1;
    else
        return path;
}

VOID RecordMemRead(ADDRINT ip, void * addr)
{
    outFile << str_of_func_at[ip] << "; " << str_of_ins_at[ip] << "; " << ip << ": R " << addr << endl;
}

// Print a memory write record
VOID RecordMemWrite(ADDRINT ip, void * addr)
{
    outFile << str_of_func_at[ip] << "; " << str_of_ins_at[ip] << "; " << ip << ": W " << addr << endl;
}

// Pin calls this function every time a new rtn is executed
VOID Routine(RTN rtn, VOID *v)
{
    string image_name = StripPath(IMG_Name(SEC_Img(RTN_Sec(rtn))).c_str());

    // if (strcmp(image_name.c_str(), TARGET_IMG.c_str()) == 0)
    if (true)
    {
        RTN_Open(rtn);
        string rtn_name = StripPath(RTN_Name(rtn).c_str());

        if (true)
        {
            for (INS ins = RTN_InsHead(rtn); INS_Valid(ins); ins = INS_Next(ins))
            {
                str_of_ins_at[INS_Address(ins)] = INS_Disassemble(ins);
                str_of_func_at[INS_Address(ins)] = rtn_name;
                // Insert a call to docount to increment the instruction counter for this rtn
                //INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)docount, IARG_PTR, &(rc->_icount), IARG_END);
                UINT32 memOperands = INS_MemoryOperandCount(ins);

                // Iterate over each memory operand of the instruction.
                for (UINT32 memOp = 0; memOp < memOperands; memOp++)
                {
                    if (INS_MemoryOperandIsRead(ins, memOp))
                    {
                        // outFile << rtn_name << " ";
                        INS_InsertPredicatedCall(
                            ins, IPOINT_BEFORE, (AFUNPTR)RecordMemRead,
                            IARG_INST_PTR,
                            IARG_MEMORYOP_EA, memOp,
                            IARG_END);
                    }
                    // Note that in some architectures a single memory operand can be 
                    // both read and written (for instance incl (%eax) on IA-32)
                    // In that case we instrument it once for read and once for write.
                    if (INS_MemoryOperandIsWritten(ins, memOp))
                    {
                        // outFile << rtn_name << " ";
                        INS_InsertPredicatedCall(
                            ins, IPOINT_BEFORE, (AFUNPTR)RecordMemWrite,
                            IARG_INST_PTR,
                            IARG_MEMORYOP_EA, memOp,
                            IARG_END);
                    }
                }
            }
        }

        RTN_Close(rtn);
    }
}

KNOB<string> KnobOutputFile(KNOB_MODE_WRITEONCE, "pintool",
    "o", "mem_access.out", "specify output file name");

// This function is called when the application exits
// It prints the name and count for each procedure
VOID Fini(INT32 code, VOID *v)
{
    outFile.setf(ios::showbase);
    outFile << "#eof" << endl;
    outFile.close();
}
/* ===================================================================== */
/* Print Help Message                                                    */
/* ===================================================================== */
INT32 Usage()
{
    cerr << "Trace" << endl;
    cerr << endl << KNOB_BASE::StringKnobSummary() << endl;
    return -1;
}
/* ===================================================================== */
/* Main                                                                  */
/* ===================================================================== */
int main(int argc, char * argv[])
{
    // Initialize symbol table code, needed for rtn instrumentation
    PIN_InitSymbols();
    // Initialize pin
    if (PIN_Init(argc, argv)) return Usage();

    outFile.open(KnobOutputFile.Value().c_str());
    // Register Routine to be called to instrument rtn
    RTN_AddInstrumentFunction(Routine, 0);
    // Register Fini to be called when the application exits
    PIN_AddFiniFunction(Fini, 0);
    
    // Start the program, never returns
    PIN_StartProgram();
    
    return 0;
}