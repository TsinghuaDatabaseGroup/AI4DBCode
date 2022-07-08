package test.agent;

import java.lang.instrument.Instrumentation;

/**
 * @author zzz_jq
 * @description TODO
 * @create 2020/12/4 18:16
 */

public class SimpleAgent {

    public static void premain(String agentArgs, Instrumentation instrumentation) throws Exception {
        instrumentation.addTransformer(new MyTransformer(), true);
        System.out.println("**************premain done!*******************");
    }

}
