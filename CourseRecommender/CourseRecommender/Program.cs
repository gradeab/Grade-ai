using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using System.IO;
using Microsoft.ML;
using Microsoft.ML.Trainers;

using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Data;

namespace CourseRecommender
{
    class Program
    {
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            (IDataView trainingDataView, IDataView testDataView) = LoadData(mlContext);

            ITransformer model = BuildAndTrainModel(mlContext, trainingDataView);

            EvaluateModel(mlContext, testDataView, model);

            Console.ReadLine();
        }
        public static (IDataView training, IDataView test) LoadData(MLContext mlContext)
        {

            //var dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "ratings.xlsx");
            var dataPath = @"c:\users\astekm\documents\exjobb\grade-ai\courserecommender\CourseRecommender\Data\recommendation-ratings-train.txt";

            IDataView dataView = mlContext.Data.LoadFromTextFile<MovieRating>(dataPath, hasHeader: true, separatorChar: ',');

            var loadedDataEnumerable = mlContext.Data
                .CreateEnumerable<MovieRating>(dataView, reuseRowObject: false);

            foreach (MovieRating row in loadedDataEnumerable)
            {
                Console.WriteLine($"{row.Label}, {row.userId}, {row.movieId}");
            }
            //var table = dataView.ToDataTable();

            //split the data into test and training
            DataOperationsCatalog.TrainTestData splitData = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.3, seed: 1);
            IDataView trainingDataView = splitData.TrainSet;
            IDataView testDataView = splitData.TestSet;


            //return (trainSet, testSet);

           

            return (trainingDataView, testDataView);
        }


        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView trainingDataView)
        {
            IEstimator<ITransformer> estimator = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "userIdEncoded", inputColumnName: "userId")
                .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "movieIdEncoded", inputColumnName: "movieId"));

            var options = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = "userIdEncoded",
                MatrixRowIndexColumnName = "movieIdEncoded",
                LabelColumnName = "Label",
                NumberOfIterations = 20,
                ApproximationRank = 100
            };

            var trainerEstimator = estimator.Append(mlContext.Recommendation().Trainers.MatrixFactorization(options));

            Console.WriteLine("=============== Training the model ===============");
            ITransformer model = trainerEstimator.Fit(trainingDataView);

            return model;
        }

        public static void EvaluateModel(MLContext mlContext, IDataView testDataView, ITransformer model)
        {
            Console.WriteLine("=============== Evaluating the model ===============");
            var prediction = model.Transform(testDataView);

            var metrics = mlContext.Regression.Evaluate(prediction, labelColumnName: "Label", scoreColumnName: "Score");

            Console.WriteLine("Root Mean Squared Error : " + metrics.RootMeanSquaredError.ToString());
            Console.WriteLine("RSquared: " + metrics.RSquared.ToString());
        }


    }

    public static class DataViewHelper
    {
        public static DataTable ToDataTable(this IDataView dataView)
        {
            DataTable dt = null;
            if (dataView != null)
            {
                dt = new DataTable();
                var preview = dataView.Preview();
                dt.Columns.AddRange(preview.Schema.Select(x => new DataColumn(x.Name)).ToArray());
                foreach (var row in preview.RowView)
                {
                    var r = dt.NewRow();
                    foreach (var col in row.Values)
                    {
                        r[col.Key] = col.Value;
                    }
                    dt.Rows.Add(r);

                }
            }
            return dt;
        }
    }
}


