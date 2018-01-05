# IssueCNTKTensorIIS

A.
CMD Project - WORKS
ArtificialIntelligence.API.CNTK

Repro: Execute console project


B. 
Web / IIS Express based
ArtificialIntelligence.API.CNTK.Web

Repro: Execute project (VS Ctrl+F5)
Go to http://localhost:49841/swagger/ui/index#!/Models/Models_AnalyzeFixedImageWithCNTK, then click on "Try it out!"

You get this error/exception:

System.StackOverflowException
  HResult=0x800703E9
  Source=<Cannot evaluate the exception source>
  StackTrace:
<Cannot evaluate the exception stack trace>
