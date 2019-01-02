package com.loring.bayes;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import java.io.IOException;
import java.util.StringTokenizer;

/**
 * @author XYL
 * @date 2018.12.28
 * 贝叶斯分类器
 */
public class BayesClassify {

    public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
        Configuration conf = new Configuration();
        FileSystem hdfs = FileSystem.get(conf);

        Path outputPath1 = new Path(args[1]);
        if(hdfs.exists(outputPath1))
            hdfs.delete(outputPath1, true);

        Job job1 =Job.getInstance(conf, "DocNumbers");
        job1.setJarByClass(BayesClassify.class);
        //设置输入输出格式
        job1.setInputFormatClass(WholeFileInputFormat.class);
        job1.setMapperClass(DocNums_Map.class);
        job1.setCombinerClass(DocNums_Reduce.class);
        job1.setReducerClass(DocNums_Reduce.class);

        FileInputFormat.setInputDirRecursive(job1,true);
        job1.setOutputKeyClass(Text.class);//reduce阶段的输出的key
        job1.setOutputValueClass(IntWritable.class);//reduce阶段的输出的value
        FileInputFormat.addInputPath(job1, new Path(args[0]));
        FileOutputFormat.setOutputPath(job1, new Path(args[1]));
        boolean isSuccess = job1.waitForCompletion(true);
        if(!isSuccess) {
            System.exit(1);
        }

        Path outputPath2 = new Path(args[2]);
        if(hdfs.exists(outputPath2))
            hdfs.delete(outputPath2, true);
        Job job2 =Job.getInstance(conf, "WordCount");
        job2.setJarByClass(BayesClassify.class);

        job2.setMapperClass(WordCount_Map.class);
        job2.setCombinerClass(WordCount_Reduce.class);
        job2.setReducerClass(WordCount_Reduce.class);

        FileInputFormat.setInputDirRecursive(job2,true);
        job2.setOutputKeyClass(Text.class);//reduce阶段的输出的key
        job2.setOutputValueClass(IntWritable.class);//reduce阶段的输出的value
        FileInputFormat.addInputPath(job2, new Path(args[0]));
        FileOutputFormat.setOutputPath(job2, new Path(args[2]));
        System.exit(job2.waitForCompletion(true) ? 0 : 1);

    }


    /*
     * 第一个MapReduce用于统计每个类对应的文件数量
     * 为计算先验概率准备:
     * 输入:args[0],训练集
     * 输出:args[1],key为类名,value为类对应的文档数目,即<ClassName,DocNums>
     */
    public static class DocNums_Map extends Mapper<NullWritable, BytesWritable, Text, IntWritable> {
        private Text newKey = new Text();
        private final static IntWritable one = new IntWritable(1);
        public void map(NullWritable key, BytesWritable value, Context context) throws IOException, InterruptedException{
            //得到当前所处理分片
            InputSplit inputsplit = context.getInputSplit();
            //将当前所处理分片的路径名按照目录结构解析为：类名、文档名
            String className = ((FileSplit)inputsplit).getPath().getParent().getName();
            //将当前所处理分片所属的类名和文档名中间加上制表符组合成一个字符串
            //String classAndDoc = className;
            newKey.set(className);
            context.write(newKey, one);
        }
    }

    public static class DocNums_Reduce extends Reducer<Text, IntWritable, Text, IntWritable> {
        private IntWritable result = new IntWritable();
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException{
            int sum = 0;
            for(IntWritable value:values){
                sum += value.get();
            }
            result.set(sum);
            context.write(key, result);
        }
    }

    public static class WholeFileInputFormat extends FileInputFormat<NullWritable, BytesWritable>{

        @Override
        protected boolean isSplitable(JobContext context, Path filename) {
            return false;
        }

        @Override
        public RecordReader<NullWritable, BytesWritable> createRecordReader(InputSplit inputSplit, TaskAttemptContext taskAttemptContext) throws IOException, InterruptedException {
            WholeFileRecordReader reader = new WholeFileRecordReader();
            reader.initialize(inputSplit, taskAttemptContext);
            return reader;
        }

    }

    public static class WholeFileRecordReader extends RecordReader<NullWritable, BytesWritable> {
        private FileSplit fileSplit;           //保存输入的分片，它将被转换成一条（key，value）记录
        private Configuration conf;     //配置对象
        private BytesWritable value = new BytesWritable(); //value对象，内容为空
        private boolean processed = false;   //布尔变量记录记录是否被处理过


        @Override
        public void initialize(InputSplit split, TaskAttemptContext context)
                throws IOException, InterruptedException {
            this.fileSplit = (FileSplit) split;  	     //将输入分片强制转换成FileSplit
            this.conf = context.getConfiguration();  //从context获取配置信息
        }
        @Override
        public NullWritable getCurrentKey() throws IOException, InterruptedException {
            return NullWritable.get();
        }
        @Override
        public BytesWritable getCurrentValue() throws IOException, InterruptedException {
            return value;
        }
        @Override
        public boolean nextKeyValue() throws IOException, InterruptedException {
            if (!processed) {  //如果记录没有被处理过
                //从fileSplit对象获取split的字节数，创建byte数组contents
                byte[] contents = new byte[(int) fileSplit.getLength()];
                Path file = fileSplit.getPath(); //从fileSplit对象获取输入文件路径
                FileSystem fs = file.getFileSystem(conf);  //获取文件系统对象
                FSDataInputStream in = null;  //定义文件输入流对象
                try {
                    in = fs.open(file);   //打开文件，返回文件输入流对象
                    IOUtils.readFully(in, contents, 0, contents.length); //从输入流读取所有字节到contents
                    value.set(contents, 0, contents.length); //将contens内容设置到value对象中
                } finally {
                    IOUtils.closeStream(in);   //关闭输入流
                }
                processed = true;   //将是否处理标志设为true，下次调用该方法会返回false
                return true;
            }
            return false;   //如果记录处理过，返回false，表示split处理完毕
        }

        @Override
        public float getProgress() throws IOException {
            return processed ? 1.0f : 0.0f;
        }

        @Override
        public void close() throws IOException {
            // do nothing
        }

    }

    /*
     * 第二个MapReduce用于统计每个类下单词的数量
     * 输入:args[0],训练集,输入为<行偏移量，单词>
     * 输出:args[2],输出为<类名_单词名，数量>
     */
    public static class WordCount_Map extends Mapper<LongWritable, Text, Text, IntWritable>{
        private Text nameAndWord = new Text();                             // KEYOUT
        private final static IntWritable one = new IntWritable(1);  // VALUEOUT
        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            InputSplit inputsplit = context.getInputSplit();
            String className = ((FileSplit)inputsplit).getPath().getParent().getName();
            String cAndTValue;
            String lineValue = value.toString();
            // 分词：将每行的单词进行分割,按照"  \t\n\r\f"(空格、制表符、换行符、回车符、换页)进行分割
            StringTokenizer tokenizer = new StringTokenizer(lineValue);
            // 遍历
            while (tokenizer.hasMoreTokens()) {
                //获取每个单词
                String wordValue = tokenizer.nextToken();
                // 设置map输出的key值为类名和单词中间加上制表符组合成的字符串
                cAndTValue = className + '\t' + wordValue;
                //将类名单词字符串的值赋给hadoop的Text对象
                nameAndWord.set(cAndTValue);
                //将<类名单词，1>键值对写入上下文
                context.write(nameAndWord, one);
            }
        }
    }

    //将相同的类名_单词累加
    public static class WordCount_Reduce extends Reducer<Text, IntWritable, Text, IntWritable> {
        private IntWritable result = new IntWritable();
        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);
        }
    }

}
