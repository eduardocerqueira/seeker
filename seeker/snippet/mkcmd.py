#date: 2022-09-07T17:03:23Z
#url: https://api.github.com/gists/649f0df183f8de498b9f6e92d3af2328
#owner: https://api.github.com/users/amuramatsu

#! /usr/bin/env python3

import os
import sys
import gzip
import base64

EXCLUDES = [ "cmd" ]

cmddir = os.path.realpath(os.path.dirname(__file__))
basedir = os.path.realpath(os.path.join(cmddir, ".."))
bindir = os.path.realpath(os.path.join(basedir, "usr", "bin"))

def make_exefile(name):
    name = os.path.basename(name)
    if name in EXCLUDES:
        return
    cmdfile = os.path.join(cmddir, "{:s}.cmd".format(name))
    exefile = os.path.join(cmddir, "{:s}.exe".format(name))
    if os.path.exists(cmdfile):
        os.remove(cmdfile)
    with open(exefile, "wb") as f:
        f.write(get_mingw_wrapper())
        
def find_shebang(dirname):
    results = []
    for f in os.listdir(dirname):
        filename = os.path.join(dirname, f)
        if f[0] == "." or os.path.isdir(filename):
            continue
        try:
            with open(filename, "rb") as f:
                firstline = f.readline()
            if firstline.startswith(b"#!"):
                results.append(filename)
        except:
            pass
    return results

_MINGW_WRAPPER_b64 = '''
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
AD4AAA==
'''
_MINGW_WRAPPER = None
def get_mingw_wrapper():
    global _MINGW_WRAPPER
    if not _MINGW_WRAPPER:
        _MINGW_WRAPPER = gzip.decompress(base64.b64decode(_MINGW_WRAPPER_b64))
    return _MINGW_WRAPPER

CPP_PROGRAM = r'''
#include <string>
#include <vector>
#include <windows.h>
#include <tchar.h>

#pragma comment(lib, "shell32.lib")
#pragma optimize("gs", on)
#pragma optimize("y", off)
#pragma runtime_checks("scu", off)

void
ArgvQuote (const std::wstring& Argument,
	   std::wstring& CommandLine,
	   bool Force)
/*++
    
Routine Description:
    
    This routine appends the given argument to a command line such
    that CommandLineToArgvW will return the argument string unchanged.
    Arguments in a command line should be separated by spaces; this
    function does not add these spaces.
    
Arguments:
    
    Argument - Supplies the argument to encode.

    CommandLine - Supplies the command line to which we append the encoded
                  argument string.

    Force - Supplies an indication of whether we should quote
            the argument even if it does not contain any characters that would
            ordinarily require quoting.
    
Return Value:
    
    None.
    
Environment:
    
    Arbitrary.
    
--*/
    
{
    //
    // Unless we're told otherwise, don't quote unless we actually
    // need to do so --- hopefully avoid problems if programs won't
    // parse quotes properly
    //
    
    if (Force == false &&
        Argument.empty () == false &&
        Argument.find_first_of (L" \t\n\v\"") == Argument.npos) {
        CommandLine.append(Argument);
    }
    else {
        CommandLine.push_back(L'"');
        
        for (auto It = Argument.begin () ; ; ++It) {
            unsigned NumberBackslashes = 0;
        
            while (It != Argument.end () && *It == L'\\') {
                ++It;
                ++NumberBackslashes;
            }
        
            if (It == Argument.end ()) {
                //
                // Escape all backslashes, but let the terminating
                // double quotation mark we add below be interpreted
                // as a metacharacter.
                //
                
                CommandLine.append (NumberBackslashes * 2, L'\\');
                break;
            }
            else if (*It == L'"') {

                //
                // Escape all backslashes and the following
                // double quotation mark.
                //
                
                CommandLine.append(NumberBackslashes * 2 + 1, L'\\');
                CommandLine.push_back(*It);
            }
            else {
                
                //
                // Backslashes aren't special here.
                //
                
                CommandLine.append(NumberBackslashes, L'\\');
                CommandLine.push_back(*It);
            }
        }
        CommandLine.push_back(L'"');
    }
}

int
main()
{
    // 自力解析
    int argc;
    LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);

    const wchar_t* mingw_path = _wgetenv(L"MINGW_PATH");
    
    if (mingw_path == NULL) {
        wchar_t path[MAX_PATH*4];
        GetModuleFileNameW(NULL, path, sizeof(path)/sizeof(path[0]));
        std::wstring pathname = path;

	//if (pathname.substr(0, 4) != L"\\\\?\\") {
	//    pathname.insert(0, L"\\\\?\\");
	//}
	pathname += L"\\..\\..";
	DWORD len = GetFullPathNameW(pathname.c_str(),
				     0, NULL, NULL);
	wchar_t* buffer = new wchar_t[len];
	GetFullPathNameW(pathname.c_str(), len, buffer, NULL);
	mingw_path = buffer;
    }
    
    std::wstring programName(argv[0]);
    std::wstring basename;
    int pos = programName.rfind(L"\\");
    if (pos == programName.length())
	basename = programName;
    else
	basename = programName.substr(pos+1);
    if (basename.length() > 4) {
        std::wstring ext = basename.substr(basename.length()-4);
        if (ext == L".exe" || ext == L".com")
	    basename = basename.substr(0, basename.length()-4);
    }
    
    std::wstring commandLine;
    std::wstring processCommand(mingw_path);
    processCommand += L"\\usr\\bin\\env.exe";
    std::wstring scriptPath(mingw_path);
    scriptPath += L"\\usr\\bin\\" + basename;

    ArgvQuote(processCommand, commandLine, false);
    commandLine += L" ";
    ArgvQuote(scriptPath, commandLine, false);
    for (int i=1; i < argc; i++) {
	commandLine += L" ";
	ArgvQuote(argv[i], commandLine, false);
    }
    //wprintf(L"CMD: %ls\n", commandLine.c_str());
    
    STARTUPINFOW si = { sizeof(STARTUPINFOW) };
    PROCESS_INFORMATION pi = {};
    if (!CreateProcessW(processCommand.c_str(),
			const_cast<wchar_t*>(commandLine.c_str()),
			NULL,
			NULL,
			TRUE, // inherit handles
			0,
			NULL,
			NULL,
			&si, &pi)) {
	printf("ERROR CreateProcess\n");
	return -1;
    }

    HANDLE childProcess = pi.hProcess;
    if (!CloseHandle(pi.hThread)) {
	printf("Error CloseHandle\n");
	return -1;
    }

    DWORD r = WaitForSingleObject(childProcess, INFINITE);
    switch (r) {
    case WAIT_FAILED:
    case WAIT_ABANDONED:
	printf("Error WaitForSingleObject\n");
	return -1;
    case WAIT_OBJECT_0:
	break;
    }
    
    DWORD exitCode;
    if (!GetExitCodeProcess(childProcess, &exitCode)) {
	printf("Error GetExitCodeProcess\n");
	return -1;
    }
    return (int)exitCode;
}
'''

def build_cpp(cpp_prog, outfile=None):
    import subprocess, gzip, base64, tempfile, shutil, textwrap

    def onerror(func, path, exc_info):
        """
        Error handler for ``shutil.rmtree``.

        If the error is due to an access error (read only file)
        it attempts to add write permission and then retries.

        If the error is for another reason it re-raises the error.

        Usage : ``shutil.rmtree(path, onerror=onerror)``
        """
        import stat
        if not os.access(path, os.W_OK):
            # Is the error an access error ?
            os.chmod(path, stat.S_IWUSR)
            func(path)
        else:
            raise
    
    tmpdir = tempfile.mkdtemp()
    try:
        CL_OPTS =  "/nologo /Oi /Ox /Os /Oy /EHsc /GF /arch:IA32".split()
        LINK_OPTS = "/MD /link shell32.lib".split()

        tmpfile = os.path.join(tmpdir, "wrapper.cpp")
        objfile = os.path.join(tmpdir, "wrapper.obj")
        exefile = os.path.join(tmpdir, "wrapper.exe")

        with open(tmpfile, "wt", encoding="utf-8") as f:
            f.write(cpp_prog)
        
        subprocess.call(
            [ "cl" ] + CL_OPTS +
            [ "/Fe" + exefile, "/Fo" + objfile, "/utf-8", tmpfile ] +
            LINK_OPTS)

        with open(exefile, "rb") as f:
            exe = f.read()
    finally:
        shutil.rmtree(tmpdir, onerror=onerror)
    
    s = base64.b64encode(gzip.compress(exe)).decode("ascii")
    if outfile is None:
        print("_MINGW_WRAPPER_b64 = '''")
        print(textwrap.fill(s, width=76))
        print("'''")
    elif outfile.endswith(".exe"):
        with open(outfile, "wb") as f:
            f.write(exe)
    else:
        with open(outfile, "wt", encoding='ascii') as f:
            f.write("_MINGW_WRAPPER_b64 = '''\n")
            f.write(textwrap.fill(s, width=76))
            f.write("\n'''\n")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--build':
        output = None
        if len(sys.argv) > 2:
            output = sys.argv[2]
        build_cpp(CPP_PROGRAM, output)
    else:
        if not os.path.isdir(bindir):
            raise Exception()

        if len(sys.argv) > 1:
            for arg in sys.argv[1:]:
                make_exefile(arg)
        else:
            for arg in find_shebang(bindir):
                make_exefile(arg)
