package test.agent;

import javassist.*;
import javassist.bytecode.ClassFile;

import java.lang.instrument.ClassFileTransformer;
import java.security.ProtectionDomain;
import java.util.concurrent.ConcurrentHashMap;

/**
 * @author zzz_jq
 * @create 2020/12/4 18:20
 */
public class MyTransformer implements ClassFileTransformer {

    @Override
    public byte[] transform(ClassLoader loader, String className, Class<?> classBeingRedefined,
                            ProtectionDomain protectionDomain, byte[] classfileBuffer) {
        if(Constants.STAGE_CREATE_CLASS.equals(className)){
            return alterStageCreate(className, classfileBuffer);
        }
        if(className.contains("org/apache/spark/") &&
                (className.contains("rdd") || className.contains("mllib") ||
                className.contains("graphx") || className.contains("api"))){
            return alterAll(className, classfileBuffer);
        }
        // org/apache/spark/SparkConf
        if("org/apache/spark/SparkConf".equals(className)){
            System.out.println("===============================");
            return alterSparkConf(className, classfileBuffer);
        }
        return classfileBuffer;
    }

    /**
     * @param className
     * @param classfileBuffer
     * @return
     */
    private byte[] alterSparkConf(String className, byte[] classfileBuffer){
        System.out.println("***************match SparkConf ***************");
        try {
            ClassPool cp = ClassPool.getDefault();
            CtClass ctClass = cp.get(className.replace("/", "."));
            // ���һ����̬��Ա����
            final CtClass defClass = cp.get(ConcurrentHashMap.class.getName());
            CtField defField = new CtField(defClass, "myMap", ctClass);
            defField.setModifiers(Modifier.STATIC);
            ctClass.addField(defField, CtField.Initializer.byNew(defClass));
            CtField[] fields = ctClass.getFields();
            for (CtField f : fields) {
                System.out.println(f.getName());
                System.out.println(f.getFieldInfo());
                System.out.println(f.getModifiers());
            }
            byte[] classData = ctClass.toBytecode();
            ctClass.detach();
            return classData;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return classfileBuffer;
    }

    /**
     * @param className
     * @param classfileBuffer
     * @return
     */
    private byte[] alterStageCreate(String className, byte[] classfileBuffer){
        System.out.println("***************match create Stage ***************");
        try {
            ClassPool cp = ClassPool.getDefault();
            CtClass ctClass = cp.get(className.replace("/", "."));
            CtMethod ctMethod = ctClass.getDeclaredMethod(Constants.STAGE_CREATE_METHOD);

            System.out.println("inserting code to createStage");
            ctMethod.insertBefore(CodeToInsert.stageSwitch.replace("\"Hello world!\"",
                        "(this.id)"));

            byte[] classData = ctClass.toBytecode();
            ctClass.detach();
            return classData;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return classfileBuffer;
    }


    private byte[] alterAll(String className, byte[] classfileBuffer){
        // System.out.println("***************altering " + className + " ***************");
        try {
            ClassPool cp = ClassPool.getDefault();
            CtClass ctClass = cp.get(className.replace("/", "."));
            ClassFile classFile = ctClass.getClassFile();
            String sourceFile = classFile.getSourceFile();
            CtMethod[] methods = ctClass.getMethods();
            for(CtMethod method : methods){
                try {

                    int startLine = method.getMethodInfo().getLineNumber(0);
                    int endLine = method.getMethodInfo().getLineNumber(Integer.MAX_VALUE);
                    String key = "\""
                            + className
                            + "."
                            + method.getName()
                            + "-" + sourceFile
                            + "-" + startLine + "-" + endLine
                            + "\"";
                    method.insertAfter(CodeToInsert.log0.replace("\"Hello world!\"", key));
                }catch (CannotCompileException e){
//                    e.printStackTrace();
                }catch (Exception e){
//                    e.printStackTrace();
                }
            }

            byte[] classData = ctClass.toBytecode();
            ctClass.detach();
            return classData;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return classfileBuffer;
    }
}

