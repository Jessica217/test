import openai
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 假设 chatGptConfig 是从配置文件中加载的字典
class ChatGptConfig:
    def __init__(self, api_keys, api_host, proxy_host, proxy_port):
        self.api_keys = api_keys
        self.api_host = api_host
        self.proxy_host = proxy_host
        self.proxy_port = proxy_port

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InitChatGptConfig:
    def __init__(self, chat_gpt_config: ChatGptConfig):
        self.chat_gpt_config = chat_gpt_config
        self.openai_client = self.create_openai_client()

    def create_openai_client(self):
        try:
            # 从配置中获取 API 和代理设置
            logger.info(f"chatgpt.apiKey: {self.chat_gpt_config.api_keys}")
            logger.info(f"chatgpt.apiHost: {self.chat_gpt_config.api_host}")
            logger.info(f"chatgpt.proxyHost: {self.chat_gpt_config.proxy_host}")
            logger.info(f"chatgpt.proxyPort: {self.chat_gpt_config.proxy_port}")

            # 配置代理
            proxies = {
                "http": f"http://{self.chat_gpt_config.proxy_host}:{self.chat_gpt_config.proxy_port}",
                "https": f"http://{self.chat_gpt_config.proxy_host}:{self.chat_gpt_config.proxy_port}"
            }

            # 配置请求超时与重试策略
            session = requests.Session()
            retry = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[500, 502, 503, 504]
            )
            adapter = HTTPAdapter(max_retries=retry)
            session.mount("http://", adapter)
            session.mount("https://", adapter)

            # 配置 OpenAI 客户端
            openai.api_key = self.chat_gpt_config.api_keys
            openai.api_base = self.chat_gpt_config.api_host

            # 使用 requests 会话来设置代理
            openai.requestssession = session
            openai.proxy = proxies

            logger.info("OpenAI client initialized successfully")
            return openai

        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            raise

# 模拟从配置文件加载的配置
chat_gpt_config = ChatGptConfig(
    api_keys="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",  # 替换为你的 OpenAI API 密钥
    api_host="https://api.openai.com/v1",  # OpenAI API 主机
    proxy_host="113.31.145.144",  # 代理主机
    proxy_port=28080  # 代理端口
)

# 初始化配置
chat_gpt_configurator = InitChatGptConfig(chat_gpt_config)
openai_client = chat_gpt_configurator.openai_client

# 你可以使用 openai_client 来与 OpenAI API 交互
response = openai.Completion.create(
    engine="text-davinci-003",
    prompt="Say this is a test.",
    max_tokens=5
)

print(response)
