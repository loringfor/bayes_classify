package com.loring.bayes;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.StringTokenizer;

public class Prediction {
    public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
        Configuration conf = new Configuration();
        FileSystem hdfs = FileSystem.get(conf);

        Path outputPath1 = new Path(args[1]);
        if(hdfs.exists(outputPath1))
            hdfs.delete(outputPath1, true);
        Job job1 =Job.getInstance(conf, "Word");
        job1.setJarByClass(Prediction.class);
        job1.setMapperClass(Prediction.WordMapper.class);
        job1.setCombinerClass(Prediction.WordReducer.class);
        job1.setReducerClass(Prediction.WordReducer.class);
        FileInputFormat.setInputDirRecursive(job1,true);
        job1.setOutputKeyClass(Text.class);//reduce阶段的输出的key
        job1.setOutputValueClass(Text.class);//reduce阶段的输出的value
        FileInputFormat.addInputPath(job1, new Path(args[0]));
        FileOutputFormat.setOutputPath(job1, new Path(args[1]));
        boolean isSuccess = job1.waitForCompletion(true);
        if(!isSuccess) {
            System.exit(1);
        }

        Path outputPath2 = new Path(args[2]);
        if(hdfs.exists(outputPath2))
            hdfs.delete(outputPath2, true);
        Job job2 =Job.getInstance(conf, "Prediction");
        job2.setJarByClass(Prediction.class);
        job2.setMapperClass(Prediction.DocOfClassMap.class);
        job2.setCombinerClass(Prediction.DocOfClassReduce.class);
        job2.setReducerClass(Prediction.DocOfClassReduce.class);
        FileInputFormat.setInputDirRecursive(job2,true);
        job2.setOutputKeyClass(Text.class);//reduce阶段的输出的key
        job2.setOutputValueClass(Text.class);//reduce阶段的输出的value
        FileInputFormat.addInputPath(job2, new Path(args[1]));
        FileOutputFormat.setOutputPath(job2, new Path(args[2]));
        System.exit(job2.waitForCompletion(true) ? 0 : 1);
    }


    public static class WordMapper extends Mapper<LongWritable, Text, Text, Text>{
        private Text newKey = new Text();
        private Text newValue = new Text();
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            InputSplit inputsplit = context.getInputSplit();
            // 类名
            String className = ((FileSplit)inputsplit).getPath().getParent().getName();
            // 文档名
            String docName = ((FileSplit)inputsplit).getPath().getName();
            StringTokenizer itr = new StringTokenizer(value.toString());
            while (itr.hasMoreTokens()) {
                newKey.set(className+"\t"+docName);
                newValue.set(itr.nextToken());
                context.write(newKey,newValue);
            }
        }
    }

    public static class WordReducer extends Reducer<Text,Text,Text,Text> {
        private Text result = new Text();
        private StringBuffer stringBuffer;
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            stringBuffer=new StringBuffer();
            for(Text word:values){
                stringBuffer = stringBuffer.append(word.toString()+" ");
            }
            result.set(stringBuffer.toString());
            System.out.println("key===>"+key);
            System.out.println("value===>"+result.toString());
            context.write(key, result);
        }
    }

    /*
     * 第三个MapReduce进行贝叶斯测试
     * 输入:args[3],处理后的测试数据，测试数据格式<<class doc>,word1 word2 ...>
     *      HashMap<String,Double> classProbably先验概率
     *      HashMap<String,Double> wordsProbably条件概率
     * 输出:args[4],输出每一份文档经贝叶斯分类后所对应的类,格式为<doc,class>
     */
    public static class DocOfClassMap extends Mapper<LongWritable, Text, Text, Text> {
        public void setup(Context context)throws IOException{
            GetPriorProbably(); //先验概率
            GetConditionProbably(); //条件概率
        }
        private Text newKey = new Text();
        private Text newValue = new Text();
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException{
            // 分词：将每行的单词进行分割,按照"\t\n\r\f"(空格、制表符、换行符、回车符、换页)进行分割
            String[] result = value.toString().split("\\s");
            String docName = result[1]; // 第二个，第一个是类名
            for(Map.Entry<String, Double> entry:classProbably.entrySet()) { //外层循环遍历所有类别
                String myKey = entry.getKey();//类名
                newKey.set(docName);//新的键值的key为<文档名>
                double tempValue = Math.log(entry.getValue());//构建临时键值对的value为各概率相乘,转化为各概率取对数再相加

                for(int i=2; i<result.length; i++){
                    String tempKey = myKey + "\t" + result[i];//构建临时键值对<class_word>,在wordsProbably表中查找对应的概率
                    if(wordsProbably.containsKey(tempKey)){
                        //如果测试文档的单词在训练集中出现过，则直接加上之前计算的概率
                        tempValue += Math.log(wordsProbably.get(tempKey));
                    }
                    else{//如果测试文档中出现了新单词则加上之前计算新单词概率
                        tempValue += Math.log(wordsProbably.get(myKey));
                    }
                }
                newValue.set(myKey + "\t" + tempValue);//新的键值的value为<类名  概率>
                context.write(newKey, newValue);//一份文档遍历在一个类中遍历完毕,则将结果写入文件,即<docName,<class  probably>>
                System.out.println(newKey + "\t" +newValue);
            }
        }
    }

    public static class DocOfClassReduce extends Reducer<Text, Text, Text, Text> {
        Text newValue = new Text();
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException{
            boolean flag = false;//标记,若第一次循环则先赋值,否则比较若概率更大则更新
            String tempClass = null;
            double tempProbably = 0.0;
            for(Text value:values){
                System.out.println("value......."+value.toString());
                String[] result = value.toString().split("\\s");
                String className=result[0];
                String probably=result[1];

                if(flag != true){//循环第一次
                    tempClass = className;//value.toString().substring(0, index);
                    tempProbably = Double.parseDouble(probably);
                    flag = true;
                }else{//否则当概率更大时就更新tempClass和tempProbably
                    if(Double.parseDouble(probably) > tempProbably){
                        tempClass = className;
                        tempProbably = Double.parseDouble(probably);
                    }
                }
            }

            newValue.set(tempClass + "\t" +tempProbably);
            //newValue.set(tempClass+":"+values.iterator().next());
            context.write(key, newValue);
            System.out.println(key + "\t" + newValue);
        }
    }

    /*计算先验概率:
     * 该静态函数计算每个类的文档在总类中占的比例,即先验概率P(c)=类c下文件总数/整个训练样本的文件总数
     * 输入:对应第一个MapReduce的输出args[1]
     * 输出:得到HashMap<String,Double>存放的是<类名,概率>
     */
    private static HashMap<String, Double> classProbably = new HashMap<String, Double>();//<类别，概率>，即<class,priorProbably>

    public static HashMap<String, Double> GetPriorProbably() throws IOException {
        Configuration conf = new Configuration();
        String filePath = "/output1/part-r-00000";
        FSDataInputStream fsr = null;
        BufferedReader bufferedReader = null;
        String lineValue = null;
        double sum = 0; //文档总数量

        try {
            FileSystem fs = FileSystem.get(URI.create(filePath), conf);
            fsr = fs.open(new Path(filePath));
            bufferedReader = new BufferedReader(new InputStreamReader(fsr));
            while ((lineValue = bufferedReader.readLine()) != null) { //按行读取
                // 分词：将每行的单词进行分割,按照"  \t\n\r\f"(空格、制表符、换行符、回车符、换页)进行分割
                StringTokenizer tokenizer = new StringTokenizer(lineValue);
                String className = tokenizer.nextToken(); //类名
                String num_C_Tmp = tokenizer.nextToken(); //文档数量
                double numC = Double.parseDouble(num_C_Tmp);
                classProbably.put(className, numC);
                sum = sum + numC; //文档总数量
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (bufferedReader != null) {
                try {
                    bufferedReader.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        Iterator iterator = classProbably.entrySet().iterator();
        while (iterator.hasNext()) {
            Map.Entry entry = (Map.Entry) iterator.next();
            Object key = entry.getKey();
            double val = Double.parseDouble(entry.getValue().toString()) / sum;
            classProbably.put(key.toString(), val);
            System.out.println(classProbably.get(key));
        }
        return classProbably;
    }


    /* 计算条件概率
     * 条件概率P(tk|c)=(类c下单词tk在各个文档中出现过的次数之和+1)/（类c下单词总数+不重复的单词总数）
     * 输入:对应第二个MapReduce的输出<<class,word>,counts>
     * 输出:得到HashMap<String,Double>,即<<类名:单词>,概率>
     */
    private static HashMap<String, Double> wordsProbably = new HashMap<String, Double>();
    public static HashMap<String, Double> GetConditionProbably() throws IOException {
        String filePath = "/output2/part-r-00000";
        Configuration conf = new Configuration();
        FSDataInputStream fsr = null;
        BufferedReader bufferedReader = null;
        String lineValue = null;
        HashMap<String,Double> wordSum=new HashMap<String, Double>(); //存放的为<类名，单词总数>

        try {
            FileSystem fs = FileSystem.get(URI.create(filePath), conf);
            fsr = fs.open(new Path(filePath));
            bufferedReader = new BufferedReader(new InputStreamReader(fsr));
            while ((lineValue = bufferedReader.readLine()) != null) { //按行读取
                // 分词：将每行的单词进行分割,按照"  \t\n\r\f"(空格、制表符、换行符、回车符、换页)进行分割
                StringTokenizer tokenizer = new StringTokenizer(lineValue);
                String className = tokenizer.nextToken();
                String word =tokenizer.nextToken();
                String numWordTmp = tokenizer.nextToken();
                double numWord = Double.parseDouble(numWordTmp);
                if(wordSum.containsKey(className))
                    wordSum.put(className,wordSum.get(className)+numWord+1.0);//加1.0是因为每一次都是一个不重复的单词
                else
                    wordSum.put(className,numWord+1.0);
            }
            fsr.close();
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (bufferedReader != null) {
                try {
                    bufferedReader.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        // 现在来计算条件概率
        try {
            FileSystem fs = FileSystem.get(URI.create(filePath), conf);
            fsr = fs.open(new Path(filePath));
            bufferedReader = new BufferedReader(new InputStreamReader(fsr));
            while ((lineValue = bufferedReader.readLine()) != null) { //按行读取
                // 分词：将每行的单词进行分割,按照"  \t\n\r\f"(空格、制表符、换行符、回车符、换页)进行分割
                StringTokenizer tokenizer = new StringTokenizer(lineValue);
                String className = tokenizer.nextToken();
                String word =tokenizer.nextToken();
                String numWordTmp = tokenizer.nextToken();
                double numWord = Double.parseDouble(numWordTmp);
                String key=className+"\t"+word;
                wordsProbably.put(key,(numWord+1.0)/wordSum.get(className));
                //System.out.println(className+"\t"+word+"\t"+wordsProbably.get(key));
            }
            fsr.close();
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (bufferedReader != null) {
                try {
                    bufferedReader.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        // 对测试集中出现的新单词定义概率
        Iterator iterator = wordSum.entrySet().iterator();	//获取key和value的set
        while (iterator.hasNext()) {
            Map.Entry entry = (Map.Entry) iterator.next();	//把hashmap转成Iterator再迭代到entry
            Object key = entry.getKey();		//从entry获取key
            wordsProbably.put(key.toString(),1.0/Double.parseDouble(entry.getValue().toString()));
        }

        return wordsProbably;
    }

}
