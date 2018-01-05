using ArtificialIntelligence.API.CNTK.Services.ComputerVision.Model;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Threading.Tasks;
using System.Web;
using System.Web.Http;

namespace ArtificialIntelligence.API.CNTK.Web.Controllers
{
    [Route("api/models")]
    public class ModelsController : ApiController
    {
        public ModelsController()
        {
        }

        [HttpPost]
        [Route("analyzeImage/predict/withCNTK")]
        public async Task<IHttpActionResult> AnalyzeImageWithCNTK()
        {
            var modelPrediction = new CNTKModelPrediction(HttpContext.Current.Server.MapPath("~/App_Data"));

            // Check if the request contains multipart / form - data.
            if (!Request.Content.IsMimeMultipartContent() ||
                HttpContext.Current.Request.Files.Count != 1 ||
                HttpContext.Current.Request.Files[0].ContentLength == 0)
            {
                throw new HttpResponseException(HttpStatusCode.UnsupportedMediaType);
            }

            try
            {
                var imageFile = HttpContext.Current.Request.Files[0];

                IEnumerable<LabelConfidence> tags = null;
                using (var image = new MemoryStream())
                {
                    await imageFile.InputStream.CopyToAsync(image);
                    tags = modelPrediction.ClassifyImage(image.ToArray());
                }

                //var tags = modelPrediction.ClassifyImage(File.ReadAllBytes(Path.Combine(HttpContext.Current.Server.MapPath("~/App_Data"), "parasol.jpg")));

                return Ok(tags.Select(t => t.Label));
            }
            catch (Exception e)
            {
                return InternalServerError(e);
            }
        }

        [HttpPost]
        [Route("analyzeImage/predict/withCNTKFixed")]
        public IHttpActionResult AnalyzeFixedImageWithCNTK()
        {
            var modelPrediction = new CNTKModelPrediction(HttpContext.Current.Server.MapPath("~/App_Data"));

            try
            {
                var tags = modelPrediction.ClassifyImage(File.ReadAllBytes(Path.Combine(HttpContext.Current.Server.MapPath("~/App_Data"), "parasol.jpg")));

                return Ok(tags.Select(t => t.Label));
            }
            catch (Exception e)
            {
                return InternalServerError(e);
            }
        }
    }
}