package edu.illinois.quicksel.basic;


public class HelloWorld {
	
	 private native void print();
	 
     public static void main(String[] args) {
//    	 System.out.println(System.getProperty("java.library.path"));
         new HelloWorld().print();
     }
     
     static {
         System.load("/Users/pyongjoo/workspace/crumbs/test/java/libhelloworld.jnilib");
//         System.loadLibrary("helloworld");
     }
}
