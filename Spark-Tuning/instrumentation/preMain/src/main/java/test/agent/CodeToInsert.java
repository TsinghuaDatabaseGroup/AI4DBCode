package test.agent;
/**
 * @author zzz_jq
 * @description TODO
 * @create 2021/6/21 18:51
 */
public class CodeToInsert {

    public static void main(String[] args) {
        //
        try {
            Class clazz = Class.forName("org.apache.spark.SparkConf");
            java.lang.reflect.Field field = clazz.getDeclaredField("myMap");
            field.setAccessible(true);
            if (java.lang.reflect.Modifier.isStatic(field.getModifiers())) {
                System.out.println(field.getName() + " , " + field.get(clazz));
            }
            java.util.concurrent.ConcurrentHashMap myMap = (java.util.concurrent.ConcurrentHashMap) field.get(clazz);
            if (myMap == null) {
                myMap = new java.util.concurrent.ConcurrentHashMap(512);
            }
            Object oldV = myMap.getOrDefault("Hello world!", new Integer(0));
            int newV = Integer.parseInt(oldV.toString()) + 1;
            myMap.put("Hello world!", new Integer(newV));
            System.out.println(myMap.size());
        } catch (Exception e) {
            e.printStackTrace();
        }

        //
        java.io.File newFile = new java.io.File("/home/jqzhuang/inst_log/inst_" + "Hello world!" + ".log");
        java.io.FileWriter writer = null;
        java.io.BufferedWriter bw = null;
        try {
            Class clazz = Class.forName("org.apache.spark.SparkConf");
            java.lang.reflect.Field field = clazz.getDeclaredField("myMap");
            field.setAccessible(true);
            if (java.lang.reflect.Modifier.isStatic(field.getModifiers())) {
                System.out.println(field.getName() + " , " + field.get(clazz));
            }
            java.util.concurrent.ConcurrentHashMap myMap = (java.util.concurrent.ConcurrentHashMap) field.get(clazz);
            field.set(null, new java.util.concurrent.ConcurrentHashMap(512));
            java.io.FileOutputStream out = new java.io.FileOutputStream(newFile);
            java.io.ObjectOutputStream objwrite = new java.io.ObjectOutputStream(out);
            objwrite.writeObject(myMap);
            objwrite.flush();
            objwrite.close();
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try{
                if (bw != null) {
                    bw.close();
                }
                if (writer != null) {
                    writer.close();
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    static String log0 = "try {\n" +
            "            Class clazz = Class.forName(\"org.apache.spark.SparkConf\");\n" +
            "            java.lang.reflect.Field field = clazz.getDeclaredField(\"myMap\");\n" +
            "            field.setAccessible(true);\n" +
//            "            if (java.lang.reflect.Modifier.isStatic(field.getModifiers())) {\n" +
//            "                System.out.println(field.getName() + \" , \" + field.get(clazz));\n" +
//            "            }\n" +
            "            java.util.concurrent.ConcurrentHashMap myMap = (java.util.concurrent.ConcurrentHashMap) field.get(clazz);\n" +
            "            Object oldV = myMap.getOrDefault(\"Hello world!\", new Integer(0));\n" +
            "            int newV = Integer.parseInt(oldV.toString()) + 1;\n" +
            "            myMap.put(\"Hello world!\", new Integer(newV));\n" +
//            "            System.out.println(myMap.size());\n" +
            "        } catch (Exception e) {\n" +
            "            e.printStackTrace();\n" +
            "        }";

    static String stageSwitch = "java.io.File newFile = new java.io.File(\"/home/jqzhuang/inst_log/inst_\" + \"Hello world!\" + \".log\");\n" +
            "        java.io.FileWriter writer = null;\n" +
            "        java.io.BufferedWriter bw = null;\n" +
            "        try {\n" +
            "            Class clazz = Class.forName(\"org.apache.spark.SparkConf\");\n" +
            "            java.lang.reflect.Field field = clazz.getDeclaredField(\"myMap\");\n" +
            "            field.setAccessible(true);\n" +
//            "            if (java.lang.reflect.Modifier.isStatic(field.getModifiers())) {\n" +
//            "                System.out.println(field.getName() + \" , \" + field.get(clazz));\n" +
//            "            }\n" +
            "            java.util.concurrent.ConcurrentHashMap myMap = (java.util.concurrent.ConcurrentHashMap) field.get(clazz);\n" +
            "            field.set(null, new java.util.concurrent.ConcurrentHashMap(512));\n" +
            "            java.io.FileOutputStream out = new java.io.FileOutputStream(newFile);\n" +
            "            java.io.ObjectOutputStream objwrite = new java.io.ObjectOutputStream(out);\n" +
            "            objwrite.writeObject(myMap);\n" +
            "            objwrite.flush();\n" +
            "            objwrite.close();\n" +
            "        } catch (Exception e) {\n" +
            "            e.printStackTrace();\n" +
            "        } finally {\n" +
            "            try{\n" +
            "                if (bw != null) {\n" +
            "                    bw.close();\n" +
            "                }\n" +
            "                if (writer != null) {\n" +
            "                    writer.close();\n" +
            "                }\n" +
            "            } catch (Exception e) {\n" +
            "                e.printStackTrace();\n" +
            "            }\n" +
            "        }";


}
