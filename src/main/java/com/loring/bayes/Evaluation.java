package com.loring.bayes;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.ArrayList;

public class Evaluation {

    public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
        Configuration conf = new Configuration();
        FileSystem hdfs = FileSystem.get(conf);

        Path outputPath1 = new Path(args[1]);
        if(hdfs.exists(outputPath1))
            hdfs.delete(outputPath1, true);

        Job job1 =Job.getInstance(conf, "OriginalDocOfClass");
        job1.setJarByClass(Evaluation.class);
        job1.setMapperClass(Evaluation.OriginalDocOfClassMap.class);
        job1.setCombinerClass(Evaluation.Reduce.class);
        job1.setReducerClass(Evaluation.Reduce.class);
        FileInputFormat.setInputDirRecursive(job1,true);
        job1.setOutputKeyClass(Text.class);//reduce阶段的输出的key
        job1.setOutputValueClass(Text.class);//reduce阶段的输出的value
        FileInputFormat.addInputPath(job1, new Path(args[0]));
        FileOutputFormat.setOutputPath(job1, new Path(args[1]));
        boolean isSuccess = job1.waitForCompletion(true);
        if(!isSuccess) {
            System.exit(1);
        }


        Path outputPath2 = new Path(args[3]);
        if(hdfs.exists(outputPath2))
            hdfs.delete(outputPath2, true);
        Job job2 =Job.getInstance(conf, "ClassifiedDocOfClass");
        job2.setJarByClass(Evaluation.class);
        job2.setMapperClass(Evaluation.ClassifiedDocOfClassMap.class);
        job2.setCombinerClass(Evaluation.Reduce.class);
        job2.setReducerClass(Evaluation.Reduce.class);
        FileInputFormat.setInputDirRecursive(job2,true);
        job2.setOutputKeyClass(Text.class);//reduce阶段的输出的key
        job2.setOutputValueClass(Text.class);//reduce阶段的输出的value
        FileInputFormat.addInputPath(job2, new Path(args[2]));
        FileOutputFormat.setOutputPath(job2, new Path(args[3]));
        //System.exit(job2.waitForCompletion(true) ? 0 : 1);
        isSuccess = job2.waitForCompletion(true);
        if(!isSuccess) {
            System.exit(1);
        }

        GetEvaluation(conf, args[1]+"/part-r-00000", args[3]+"/part-r-00000");

    }


    /**
     * 得到原本的文档分类
     * 输入:初始数据集合,格式为<<ClassName Doc>,word1 word2...>
     * 输出:原本的文档分类，即<ClassName,Doc>
     */
    public static class OriginalDocOfClassMap extends Mapper<LongWritable, Text, Text, Text> {
        private Text newKey = new Text();
        private Text newValue = new Text();

        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException{
            // 分词：将每行的单词进行分割,按照"\t\n\r\f"(空格、制表符、换行符、回车符、换页)进行分割
            String[] result = value.toString().split("\\s");
            String className = result[0]; // 类名
            String docName = result[1]; // 文档名
            newKey.set(className);
            newValue.set(docName);
            context.write(newKey, newValue);
            System.out.println(newKey + "\t" + newValue);
        }
    }

    /**
     * 得到经贝叶斯分分类器分类后的文档分类
     * 读取经贝叶斯分类器分类后的结果文档<Doc,ClassName 概率>,并将其转化为<ClassName,Doc>形式
     */
    public static class ClassifiedDocOfClassMap extends Mapper<LongWritable, Text, Text, Text>{
        private Text newKey = new Text();
        private Text newValue = new Text();
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException{
            // 分词：将每行的单词进行分割,按照"\t\n\r\f"(空格、制表符、换行符、回车符、换页)进行分割
            String[] result = value.toString().split("\\s");
            String docName = result[0]; // 文档名
            String className = result[1]; // 类名
            newKey.set(className);
            newValue.set(docName);
            context.write(newKey, newValue);
        }
    }

    public static class Reduce extends Reducer<Text, Text, Text, Text> {
        private Text result = new Text();
        public void reduce(Text key, Iterable<Text>values, Context context) throws IOException, InterruptedException{
            //生成文档列表
            StringBuffer fileList = new StringBuffer();
            for(Text value:values){
                fileList.append(value + "\t");
            }
            result.set(fileList.toString());
            context.write(key, result);
        }
    }



    /**
     * 第一个MapReduce计算得出初始情况下各个类有哪些文档,第二个MapReduce计算得出经贝叶斯分类后各个类有哪些文档
     * 此函数作用就是统计初始情况下的分类和贝叶斯分类两种情况下各个类公有的文档数目(即针对各个类分类正确的文档数目TP)
     * 初始情况下的各个类总数目减去分类正确的数目即为原本正确但分类错误的数目(FN = OriginalCounts-TP)
     * 贝叶斯分类得到的各个类的总数目减去分类正确的数目即为原本不属于该类但分到该类的数目(FP = ClassifiedCounts - TP)
     */
    //Precision精度:P = TP/(TP+FP)
    //Recall精度:   R = TP/(TP+FN)
    //P和R的调和平均:F1 = 2PR/(P+R)
    //针对所有类别:
    //Macroaveraged(宏平均) precision:(p1+p2+...+pN)/N
    //Microaveraged(微平均) precision:对应各项相加再计算总的P、R值
    public static void GetEvaluation(Configuration conf, String ClassifiedDocOfClassFilePath, String OriginalDocOfClassFilePath) throws IOException{

        //原始文档
        FileSystem fs1 = FileSystem.get(URI.create(OriginalDocOfClassFilePath), conf);
        FSDataInputStream fsr1 = fs1.open(new Path(OriginalDocOfClassFilePath));
        BufferedReader reader1 = new BufferedReader(new InputStreamReader(fsr1));
        //分类后的文档
        FileSystem fs2 = FileSystem.get(URI.create(ClassifiedDocOfClassFilePath), conf);
        FSDataInputStream fsr2 = fs2.open(new Path(ClassifiedDocOfClassFilePath));
        BufferedReader reader2 = new BufferedReader(new InputStreamReader(fsr2));

        ArrayList<String> ClassNames = new ArrayList<String>();     //依次得到分类的类名
        ArrayList<Integer> TruePositive = new ArrayList<Integer>(); // TP,记录真实情况和经分类后，正确分类的文档数目
        ArrayList<Integer> FalseNegative = new ArrayList<Integer>();// FN,记录属于该类但是没有分到该类的数目
        ArrayList<Integer> FalsePositive = new ArrayList<Integer>();// FP,记录不属于该类但是被分到该类的数目
        ArrayList<Double> precision = new ArrayList<Double>();
        ArrayList<Double> recall = new ArrayList<Double>();
        ArrayList<Double> F1 = new ArrayList<Double>();

        try{
            reader1 = new BufferedReader(new InputStreamReader(fsr1));//创建Reader对象
            String lineValue1 = null;
            String lineValue2 = null;
            while ((lineValue1 = reader1.readLine()) != null && (lineValue2 = reader2.readLine()) != null) { //按行读取
                // 分词：将每行的单词进行分割,按照"  \t\n\r\f"(空格、制表符、换行符、回车符、换页)进行分割
                String[] result1 = lineValue1.split("\\s");
                String[] result2 = lineValue2.split("\\s");
                //后面可以逐条记录处理(因为每次读入的分类前后的类是相同的)
                //System.out.println(result1[0]+"\t"+result2[0]);
                String className = result1[0];
                ClassNames.add(className);

                int TP = 0;
                for(int i=1; i<result2.length; i++){ // result2是分类文档
                    for(int j=1; j<result1.length; j++){
                        if(result2[i].equals(result1[j])){
                            TP++;
                        }
                    }
                }

                TruePositive.add(TP);
                int FP = result2.length - TP - 1;// FP = ClassifiedCounts - TP，减1是因为最开始的是类名
                int FN = result1.length - TP - 1;// FP = OriginalCounts - TP，减1是因为最开始的是类名
                FalsePositive.add(FP); // FP
                FalseNegative.add(FN); // TP
                double p = TP * 1.0 / ( TP + FP );
                double r = TP * 1.0 / ( TP + FN );
                double F = 2 * p * r / ( p + r );
                precision.add(p);
                recall.add(r);
                F1.add(F);
                System.out.println(className +"\t precision: "+ p);
                System.out.println(className +"\t recall: "+ r);
                System.out.println(className +"\t F1: "+ F);
                System.out.println();

            }

            //Caculate MacroAverage
            double precisionSum = 0.0;
            double recallSum = 0.0;
            double F1Sum = 0.0;

            //Macroaveraged(宏平均) precision:(p1+p2+...+pN)/N
            for(int i=0; i<ClassNames.size(); i++){
                precisionSum += precision.get(i);
                recallSum += recall.get(i);
                F1Sum += F1.get(i);
            }
            int n = ClassNames.size();
            System.out.println("average precision: "+ precisionSum /n );
            System.out.println("average recall: "+ recallSum / n );
            System.out.println("average F1: "+ F1Sum / n );

        }finally{
            reader1.close();
            reader2.close();
        }
    }
}