function makeExport( projectName, filename) //功能函数
{
    var project = Application.Spck.openProject(projectName);
    var dir;
    try
    {
        // build the name of the directory, where all file will be placed
        dir = new Dir(
            project.fileName.substring(
                0,
                project.fileName.length - project.baseFileName.length - 4
            )
        );
        // create the directory and make it current
        dir.setCurrent();
    }
    catch(e)
    {
        throw e;
    }


    var outfile;
    try
    {
        outfile = new File(filename+".dat");
        outfile.open(File.WriteOnly);
    }
    catch(e)
    {
        throw e;
    }

    // project name
    outfile.writeLine(project.title.val); // write some header informations to outfile

    // creation date
    var today = new Date();
    outfile.writeLine(today.toString());

    var pageSet;
    var page;
    var diagram;

    // loop over all pagesets
    for( var iPageSet = 0 ; iPageSet < project.getNumPageSets() ; iPageSet++ )
    {
        pageSet = project.getPageSet(iPageSet);
        // loop over all pages
        for( var iPage = 0 ; iPage < pageSet.getNumPages() ; iPage++ )
        {
            page = pageSet.getPage(iPage);
            // loop over all diagrams
            for( var iDiagram = 0 ; iDiagram < page.numDiagrams ; iDiagram++ )
            {
                diagram = page.getDiagram(iDiagram);
                // loop over all curves
                for(var iCurve = 0; iCurve < diagram.numCurves; iCurve++)
                {
                    var curve = diagram.getCurve(iCurve);
                    // add blank line before curve value block
                    outfile.writeLine("");
                    // write curve name & number of value pairs to outfile
                    var numValues = curve.getNumValues();
                    outfile.writeLine(curve.title.val);
                    outfile.writeLine(numValues);

                    // write value pairs to outfile
                    for( var iValuePair = 0 ; iValuePair < numValues ; iValuePair++ )
                    {
                        // created output value string, using scientific format
                        var formatX = new String("%1").argDec( 
                            curve.getXValue(iValuePair), 14 , 'E', 8 
                        );
                        var formatY = new String("%1").argDec( 
                            curve.getYValue(iValuePair), 14 , 'E', 8 
                        );
                        // write line to outfile
                        outfile.writeLine( formatX + " ; " + formatY );
                    }
                }
            }
        }
    }

    Application.Spck.closeProject(project);
    outfile.close();
}

// entry point for the batch modus
function main(args) //入口函数
{
    // Command line execution: spckgui -s SCRIPT_NAME PROJECT_NAME RESULT_NAME
    // check if there is an args variable (command line arguments)
    if(args)
    {
        //print(" args.length = " + args.length + "\n"); // debug info
        //print(" args = " + args + "\n"); // debug info

        // check if there are two command line arguments
        if(args.length == 2)
        {
            makeExport(args[0], args[1]);
        }
        else
        {
            MessageBox.critical("Call with\nspckgui -s SCRIPT_NAME PROJECT_NAME RESULT_NAME");
        }
    }
}


