package com.tophant.vip.openai.config;

import com.unfbx.chatgpt.OpenAiStreamClient;
import com.unfbx.chatgpt.function.KeyRandomStrategy;
import com.unfbx.chatgpt.interceptor.OpenAILogger;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import okhttp3.OkHttpClient;
import okhttp3.logging.HttpLoggingInterceptor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.stereotype.Component;

import java.net.InetSocketAddress;
import java.net.Proxy;
import java.util.List;
import java.util.concurrent.TimeUnit;

@Configuration
@Slf4j
@Data
public class InitChartGptConfig {

//    @Value("${chatgpt.apiKey}")
//    private List<String> apiKey;
//    //自定义key使用策略 默认随机策略
//    @Value("${chatgpt.apiHost:'https://api.openai.com/v1'}")
//    private String apiHost;
//
//    @Value("${chatgpt.proxyHost:'113.31.145.144'}")
//    private String proxyHost;
//    @Value("${chatgpt.proxyPort:28080}")
//    private Integer proxyPort;
    @Autowired
    ChatGptConfig chatGptConfig;

    @Bean
    public OpenAiStreamClient openAiStreamClient() {

        try {
            log.info("chatgpt.apiKey:{}", chatGptConfig.getApiKeys());
            log.info("chatgpt.apiHost:{}", chatGptConfig.getApiHost());
            log.info("chatgpt.proxyHost:{}", chatGptConfig.getProxyHost());
            log.info("chatgpt.proxyPort:{}", chatGptConfig.getProxyPort());

            //本地开发需要配置代理地址
            Proxy proxy = new Proxy(Proxy.Type.HTTP, new InetSocketAddress(chatGptConfig.getProxyHost(), chatGptConfig.getProxyPort()));
            HttpLoggingInterceptor httpLoggingInterceptor = new HttpLoggingInterceptor(new OpenAILogger());
            //!!!!!!测试或者发布到服务器千万不要配置Level == BODY!!!!
            //!!!!!!测试或者发布到服务器千万不要配置Level == BODY!!!!
            httpLoggingInterceptor.setLevel(HttpLoggingInterceptor.Level.HEADERS);
            OkHttpClient okHttpClient = new OkHttpClient
                    .Builder()
                    .proxy(proxy)
                    .addInterceptor(httpLoggingInterceptor)
                    .connectTimeout(30, TimeUnit.SECONDS)
                    .writeTimeout(600, TimeUnit.SECONDS)
                    .readTimeout(600, TimeUnit.SECONDS)
                    .build();
            return OpenAiStreamClient
                    .builder()
                    .apiHost(chatGptConfig.getApiHost())
                    .apiKey(chatGptConfig.getApiKeys())
                    //自定义key使用策略 默认随机策略
                    .keyStrategy(new KeyRandomStrategy())
                    .okHttpClient(okHttpClient)
                    .build();
        } catch (Exception e) {
            log.error(e.getMessage());
            e.printStackTrace();
        }
        return null;
    }

}
