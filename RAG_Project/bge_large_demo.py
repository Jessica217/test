#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce :
@File      : bge_large_demo.py
@Time      : 2024/8/13 9:06
@Author    : wei.xu
@Tel       : 
@Email     : wei.xu@tophant.com
@pip       : pip install 
"""
import sys
import os
import time
import datetime
import numpy as np
import pandas as pd
import warnings
import json
from pathlib import Path

warnings.simplefilter("ignore")
# 显示所有列
pd.set_option('display.max_columns', 20)
# 显示所有行
pd.set_option('display.max_rows', 50)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
parentPath = os.path.split(rootPath)[0]
grandFatherPath = os.path.split(parentPath)[0]
sys.path.append(curPath)
sys.path.append(rootPath)
sys.path.append(parentPath)
sys.path.append(grandFatherPath)
from FlagEmbedding import FlagModel
sentences_1 = ["你好"]
sentences_2 = ["你好","中国", "hello"]
# model_name='/data/huggingface_models/bge-reranker-v2-m3'
model_name='/data/huggingface_models/bge-large-zh-v1.5'
model = FlagModel(model_name,
                  query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                  use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
embeddings_1 = model.encode(sentences_1)
print(f"embeddings_1: {embeddings_1}")
embeddings_2 = model.encode(sentences_2)
print(f"embeddings_2: {embeddings_2}")
similarity = embeddings_1 @ embeddings_2.T
print(similarity)
print(f"similarity: {similarity}")
#
# # for s2p(short query to long passage) retrieval task, suggest to use encode_queries() which will automatically add the instruction to each query
# # corpus in retrieval task can still use encode() or encode_corpus(), since they don't need instruction
# queries = ['query_1', 'query_2']
# passages = ["样例文档-1", "样例文档-2"]
# q_embeddings = model.encode_queries(queries)
# p_embeddings = model.encode(passages)
# scores = q_embeddings @ p_embeddings.T
#
# sentences_1="""
# The Nepalese securities markets is being modernised due to some structural changes in the recent years. The fullfledged dematerialised transaction of securities, the introduction of ASBA, CASBA and Meroshare system in the primary market enabling the applicants from 77 districts to access the service through more than 2500 BFIs as service providers, branch expansion of merchant bankers and stockbrokers outside of Kathmandu valley and adoption of online trading system have made Nepalese securities markets technofriendly, investment friendly and countrywide resulting increased attraction of public towards the securities markets in recent days. Low level of participation of institutional investors in the markets, lack of diversified instruments and low level of understanding and awareness in securities markets continues to be a cause of concern
# The Nepalese securities markets is being modernised due to some structural changes in the recent years. The fullfledged dematerialised transaction of securities, the introduction of ASBA, CASBA and Meroshare system in the primary market enabling the applicants from 77 districts to access the service through more than 2500 BFIs as service providers, branch expansion of merchant bankers and stockbrokers outside of Kathmandu valley and adoption of online trading system have made Nepalese securities markets technofriendly, investment friendly and countrywide resulting increased attraction of public towards the securities markets in recent days. Low level of participation of institutional investors in the markets, lack of diversified instruments and low level of understanding and awareness in securities markets continues to be a cause of concern
# The Nepalese securities markets is being modernised due to some structural changes in the recent years. The fullfledged dematerialised transaction of securities, the introduction of ASBA, CASBA and Meroshare system in the primary market enabling the applicants from 77 districts to access the service through more than 2500 BFIs as service providers, branch expansion of merchant bankers and stockbrokers outside of Kathmandu valley and adoption of online trading system have made Nepalese securities markets technofriendly, investment friendly and countrywide resulting increased attraction of public towards the securities markets in recent days. Low level of participation of institutional investors in the markets, lack of diversified instruments and low level of understanding and awareness in securities markets continues to be a cause of concern
# The Nepalese securities markets is being modernised due to some structural changes in the recent years. The fullfledged dematerialised transaction of securities, the introduction of ASBA, CASBA and Meroshare system in the primary market enabling the applicants from 77 districts to access the service through more than 2500 BFIs as service providers, branch expansion of merchant bankers and stockbrokers outside of Kathmandu valley and adoption of online trading system have made Nepalese securities markets technofriendly, investment friendly and countrywide resulting increased attraction of public towards the securities markets in recent days. Low level of participation of institutional investors in the markets, lack of diversified instruments and low level of understanding and awareness in securities markets continues to be a cause of concern
# """
# embeddings_1 = model.encode(sentences_1)
# print(f"embeddings_1 :{embeddings_1}")
# print(f"+++++++++++++++++++++++")
# query="徐伟是谁啊"
#
# contexts=['901 20.9.46 Hyper Hasher Hyper Hasher 是一个由 Matt "Cyber Do g" LaPlante 创建的共享软件程序。 Hyper Hasher 的目标是允许这个用户计算数字文件哈希和以更多形式求校验和而不是任何其它程序。 它也 允许任何支持格式的多种文件的快速对照。 20.9.47 链接哄骗 (Hyperlink Spoofing) 超链接哄骗 (Hyperlink Spoofing) ，也叫做网页哄骗，是通过一个可能看到或使网页发生 改变的攻击者采用的手段来从一台计算机传输到另一台。 这个页面包括包括机密信息例如登 录在线的信用卡号。这个技术被用于数据网络钓鱼。 20.10 l 20.10.1 IA: Information Assu rance (信息保障 ) 信息保障 (IA)是指保证信息系统安全的方法论。 20.10.2 IAE: Information Assurance Engineering (信息保障工 程) 信息保障工程 (IAE) 通过可用性（保护免受拒绝服务攻击） ；完整性（保护免受未经修 改的数据修改） ；鉴定 (保护免受哄骗和伪造 )；机密性 (保护免受未授权的泄漏 )；和不可否认 (保护免受交易分享拒绝 )来保护和防护信息和信息系统。这包括为信息系统通过合并保护提 供恢复，探测和反应能力。对国家安全系统， IAE 是一个系统工程的重要部分，其可以部分 地符合全部的系统安全要求。 20.10.3 IASE: Information Assurance Support Environment (信息保障支持环境 ) 信息保障支持环境 (IASE)是一个美国防御部门 (DoD)信息保障 (IA)信息清算机构。', '1061 20.26.16 zoo 病毒园 zoo 病毒园是只存在于病毒和反病毒实验室中的病毒及蠕虫清单和样本， 这些样本可以 用来研究病毒代码的生成以及传染方式。 21 348个网络安全“黑话”词典 21.1 编者说明 资料原文中没有术语名称，编者根据词条 的内容，猜测补充了名称，未必准确，仅供参 考。 21.2 术语 21.2.1 Burp Suite Burp Suite 是一款信息安全从业人员必备的集成型的渗透测试工具 ，它采用自动测试和 半自动测试的方式，通过拦截 HTTP/HTTPS 的Web数据包，充当浏览器和相关应用程序的中 间人，进行拦截、修改、重放数据包进行测试，是 Web安全人员的一把必备的瑞士军刀。 21.2.2 Bypass Bypass就是绕过的意思，渗透测试人员通过特殊语句的构建或者做混淆等进行渗透测 试，然后达到绕过 WAF的手法。 21.2.3 C2 C2全称为 Command and Control ，命令与控制，常见于 APT攻击场景中。作动词解释时 理解为恶意软件与攻击者进行交互，作名词解释时理解为攻击者的“基础设施” 。 21.2.4 CC攻击 CC攻击的原理是通过代理服务器或者大量肉鸡模拟多个用户访问目标网站的动态页面， 制造大量的后台数据库查询动作，消耗目标 CPU资源，造成拒绝服务。 CC不想 DDoS可以 用硬件防火墙来过滤攻击， CC攻击本身的请求就是正常的请求。', '1121 22.2.35 CSRF CSRF英文全称 Cross Site Request Forgery ，中文为跨站站点请求伪造，简称跨站请求伪 造。它是一种挟制用户在当前已登录的 Web应用程序上执行非本意的操作的攻击方法。跟 XSS相比， XSS利用的是用 户对指定网站的信任， CSRF利用的是网站对用户网页浏览器的信 任。 22.2.36 CVE(Common Vulnerabilities & Exposures) - 漏洞编 号 CVE 的英文全称是 `Common Vulnerabilities & Exposures` 公共漏洞和暴露， CVE就好像 是 一个字典表， 为广泛认同的信息安全漏洞或者已经暴露出来的弱点给出一个公共的名称。 如果在一个漏洞报告中指明的一个漏洞，如果有 CVE名称，你就可以快速地在任何其它 CVE 兼容的数据库中找到相应修补的信息，解决安全问题。 22.2.37 CVSS CVSS英文全称是 Common Vulnerability Scoring System ，中文为通用漏洞评分系统，是一 个行业公开标准，其被设计用来评测漏洞的严重程度，并帮助确定所需反应的紧急度和重要 度。当前 CVSS V3为最新标准。 CVSS和CVE一同由美国国家漏洞库（简称 NVD）发布并保 持数据的更新。 22.2.38 Cyber attack网络攻击 故意和恶意尝试通过网络手段破坏，破坏或访问计算机系统，网络或设备。 22.2.39 Cyber incident网络事件 试图获得对系统和 /或数据的未授权访问。 /未经授权使用系统 来处理或存储数据。 /未经 系统所有者同意，更改系统的固件，软件或硬件。 /恶意破坏和 /或拒绝服务。 22.2.40 Cyber security网络安全 网络安全是一个集体术语，用于描述针对电子和计算机网络，程序和数据的保护，以防 止恶意攻击和未经授权的访问。', '240 8.20.80 Single Loss Expectancy: 单一损失期望 (SLE) 与针对特定资产的单个己发生的风险有关的成本， 表示组织发生的某种资产被特定威胁 损坏所造成的精确损失值。 SLE =资产价值 (美元 ) x 暴露因子 (EF) 。 8.20.81 single point of failure: 单点故障 基础架构的任何元素 (例如设备、 服务、 协议或通信链路 )， 如果受到破坏、 侵犯或毁坏， 将导致完全或重大的停机时间，从而影响组织成员执行基本工作任务的能力。 8.20.82 Single SignOn: 单点登录 (SSO ) 准许主体在系统上只进行一次身份认证的机制。利用 SSO，一旦主体通过身份认证，那 么他们可能在网络中自由漫游，并且不必再次接受身份认证挑战就可以访问资源和服务。 8.20.83 single state: 单一状态 要求使用策略机制来管理不同安全级别信息的系统。在这种类型的方案中，安全系统管 理员准许处理器和系统一次只处理一个安全级别的问题。 8.20.84 singleuse passwords: 专用密码 一种动态密码，每次使用时它们都会发生改变。 8.20.85 site survey: 现场勘测 使用 RF 信号检测器对无线信号的强度、质 量和干扰的正式评估。 8.20.86 Skipjack 与托管加密标准联合， 对 64 位的文本分组进行操作的算法。 它使用一个 80 位的密钥， 井且支持相同的 4 种 DES 操作模式。 Skipjack 由美国政府提议，但从未付诸实践。它提供 了支持 Clipperr 和 Capstone 高速加密芯片的密码学程序， 这些芯片是为重要商业应用而设 计的。', '842 20.4.77 因特网安全中心 (CIS: Center for Internet S ecurity) 因特网安全中心 (CIS)是一个非盈利的组织， 其帮助机构管理与信息系统安全有关的风险。 20.4.78 认 证 的 信 息 系 统 审 计 师 (CISA: Certificate Information Systems Auditor) 认证的信息系统审计师 (CISA)是一个被在审计、控制和信息系统安全中被广泛采用的认 证。 20.4.79 首席信息安全执行官 (CISO) 首席信息安全执行官 (CISO)是在公司负责所有与计算机和信息安全有关的事宜的人。首 席安全官 (CSO)是在公司内部负责维护安全的，包括员工的人身，资源和信息安全。 20.4.80 CISSP 注册信息安全专业人员 (CISSP)是一个由信息安全领域内的困际信息安全认证协会 (ISC2) 提供的一个认证程序。 CISSP 认证提供信息安全专寸门人员而仅是一个能力的目标 测量标准 而且是一个成就的全球认证标准。这个 CISSP凭据证明了在 (ISC) 2 CISSP CBK 中 10 个领城 的能力。 这个 CISSP 凭据对于正在努力或已经获得诸如 CISOs, CSOs 或高级安全工程帅的中 层和高层管理人员尤为适合。 20.4.81 CITU 中部信息技术单元 (CITU)是英国政府中的组织，负责信息技术政策及策略、设立政府 中 的相关组织、推广信息技术的使用并向大众提供政府服务。 20.4.82 ClarkWilson模式 ClarkWilson模式的开发是为了解决商业环境的安全问题，而主要是跟数据完整性相关 的问题。该模式使用两种机制来实现数据完整性：合理结构、任务分开。 20.4.83 C2（安全）级 C2安全级由美国国家计饵机安全中心 (NCSC)制定，通过了国防部 (DoD)可信计算机系统', '335 的攻击。因为一个合法网站代码中的错误，攻击者能够使该网站从攻击者的网站返回代码给 正在访问合法网站的浏览器。受害者的浏览器认为代码是从合法网站传来的，因此执行了这 些代码。 12.4.24 CSMA/CA （Carrier Sense Multiple Access/Collision Avoidance ） （载波侦听多路访问 /冲突避免） 用于 802.11 标准的无线局域网和以太网网络的第二层网络争论协议。 CSMA/CA 积极 响应每个发送帧，以避免在无线网络中的冲突。 12.4.25 CSMA/CD（Carrier Sense Multiple Access/Collision Detection ） （载波侦听多路访问 /冲突检测） 用于在有线以太网络检测同时传输的数据包（碰撞） ，并提供一种重传这些数据包能力 的第二层网络争论协议。 12.4.26 CVE（Common Vulnerabilities and Exposures ） （通用 漏洞披露） 由 MITRE 公司（一个非营利的联邦资助机构）创建并维护的通用软件所有已知安全漏 洞的数据库。该数据库中的漏洞被归类为 CWE 定义的类别。参见 CWE（通用弱点 枚举） 。 12.4.27 CWE（Common Weakness Enumeration ） （通用弱点 枚举） MITRE 列出的安全漏洞和在程序代码和编码实践中的安全弱点，如缓冲区溢出和不足 的数据验证。相较于 CVE，这个清单层次更高并更注重安全缺陷的类别。请参阅 CVE（通用 漏洞披露） 。 12.5 D 12.5.1 D2D（diskto-disk）technology （磁盘到磁盘技术） 使用磁盘阵列或设备的磁盘来存储数据。慢的磁带备份系统可能是一个瓶颈，因为服务', '369 12.20.18 SID（security identifier ） （安全标识符） Windows 中用于标识安全主体的唯一号码。参见 security principal （安全主体） 。 12.20.19 sideload（侧负载） 在移动设备上安装一个不是从官方应用程序源代码获得的应用（最常用的是 Android 设备） 。还可以指传输计算机或存储卡的数据到移动设备。 12.20.20 SIEM（Security Information and Event Management ） （安全信息与事件管理） 从计算机、网络设备和安全检测系统，如入侵检测系 统和防病毒软件收集日志、警报和 数据，并智能的关联信息的一种技术，以帮助安全专业人员做出关于威胁的明智决定。这是 该词最正确的形式。你也可能看到它写的是“应急事件” ，而不是“信息”或是“监控” ，也 不是 “管理” 。 术语 “信息” 是首选项， 因为 SIEM收集的不仅仅是事件的警报 ——包括日志、 网络流量分析以及用于关联信息的其他数据。术语“管理”是优选的，因为它暗示了更积极 的评价信息。 12.20.21 signature （签名） 识别特定威胁的预定义字节模式。 12.20.22 signaturedetection IDS （签名检测的 IDS） 通过将捕获的流量比对已知不良模式数据库的入侵检测系统。 参见 IDS（入侵检测系统） 和 HIDS（主机入侵检测系统） 。 12.20.23 site survey （现场调查） 实地考察，确定射频覆盖的轮廓和属性。 12.20.24 SLE（single loss expectancy ） （单次预期损失） 基于 ARE 的风险分析，非期望事件的预期成本。参见 ALE（年预期损失）和 ARO（年 发生概率） 。', "998 20.20.5 Sadmind Sadmind 是使一个平台妥协使之攻击另一个的蠕虫。 20.20.6 SAFE Architecture 安全架构 SAFE 构架是由 Cisco 系统研发的网络安全结构。 SAFE 试图成为在 Cisco AVVID( 语音、 影像及整合数据的整休解决方案 )下灵活多变的蓝图架构。 20.20.7 Safe Harbor 安全港协议 鉴于数字化个人数据管理执行情况的不同， 美国商业部跟欧洲联盟建立了安全港协议框 架。 加入安全港的美国企业必须单独从各个欧洲国家获取授权， 以免侵犯欧洲隐私法的条款。 20.20.8 Safe Harbor Agreement （避风港协定） 避风港协定 (Safe Harbor Agreement) 是一个关于传输个人验证信息 (PII)的国际协定协议。 20.20.9 避风港原理 (Safe Harbor Principles) 避风港原理 (Safe Harbor Principles) 在美国和欧洲联盟 (EU)之间对协调因素保护实践的一 连串指示。 20.20.10 Safety (安全 ) 安全 (Safety)是指需要保证公司有关的人，包括雇员、用户和访问者被保护免受伤害。 20.20.11 SAINT: Security Administrator's Integrated Network Tool (安全管理员集成网络工具 ) 安全管理员集成网络工具 (SAINT)是评估一个网络安全的工具。 SAINT 能够扫描网络安全 缺陷攻击和准 备详细报道这些弱点的广度和严肃性，也提供链接来确定和介绍安全程序。 SAINT最初为基于 UNIX的系统发展的，它最近已经被发展到其它的操作系统例如 Mac OS X。", '856 20.5.7 DCS1000 DCS1000，以前叫做 Carnivore , 是一个被 FBI 用于监控邮件的监视系统。 20.5.8 数据管理人 数据管理人是当前正在使用或管理数据的实体，因此该实体也会暂时地对数据负责。 20.5.9 DEA:数据加密算法 (Data Encryption Algorithm) 数据加密算法 (DEA)是被定义作为美国政府数据加密标准的一部分的对称块密码。 DEA 使用一个 64 位的密钥，其中的 56 位是独立选择的， 8 位是校验位，和映射一个 64 位块 到另一个 64 位块。 20.5.10 DES: 数据加密标准 数据加密标准 (DES)是美国的一项很长的加密标准，使用了 ANSI 利用标准进行规定的 对称密钥加密法， ANSI 于 1981 年制定了 ANSI X.3.92. DES 对此种加密方法进行了规定： 需要使用 56 位的密钥和密码块方式，即将 文本按 64 位大小分成若干份，然后对它们进行 加密。在这种算法中可以使用的密钥数为 7.2x1015。同其它私钥加密法一样，使用这种加密 方法的发送者和接收者都必须知道并能够使用这种私钥。 20.5.11 数据完整性 就数据和网络安全而言，完整性是指确保数据只能为有权的用户进行存取和修改。用于 确保数据完整性的措施包括控制联网终和服务器的物理环境、 限制对数据的访问和保持使用 严格的验证机制。 20.5.12 数据密钥 在密码学中，数据密钥是用于字符串或文本加密或解密过程中的变量值。它只对数据进 行加密或解密，而不会对其它密钥进行加密或解密，就像一 些加密规则规定的那样。 20.5.13 数据挖掘 数据挖掘是一种用于对现存信息进行分析的技术，其目的通常是为业务寻求新方法。', '905 20.10.23 IKE 因特网密钥交换是一种 IPSec(安全因特网协议 )的标准协议，可用于确保虚拟专用网络 (VPN)和远程主机或网络访问的安全。 IETF 的 2409号请求注解规定 IKE是 IPSec 安全通讯 (SA)的一种通讯和验证的方式。 20.10.24 IKEv2: Internet Key Exchange version 2 (因特网密钥 交换协议版本 2) 因特网密钥交换协议版本 2(IKEv2)，因特网交换协议 (IKE)，是一个 IPSe c(因特网协议安 全)标准协议，用于保证虚拟私有网络 (VPN)的安全流通和远程主机或网络接入。 20.10.25 ILOVEYOU ILOVEYOU ，是 Loveletter 蠕虫的另一个名称，是一个恶意的 VBScript 程序，其使用微 软 Outlook 地址名录传播。 20.10.26 IM spam 即时消息垃圾信息 即时消息垃圾信息也被叫做 spim，是通过即时消息 (IM)进行传输的一种垃圾信息。 20.10.27 IM worm 即时消息蠕虫 即时消息蠕虫是一种能够自我复制的恶意代码，可在即时消息网络中进行传播。当一种 即时消息蠕虫感染了某计算机时，它 为发送即时消息的客户端软件编制地址本，也叫做伙伴 列表或联系列表，并试图将自己向受感染计算机用户所联系过的地址进行传输。 20.10.28 Impersonation (模拟 ) 模拟 (Impersonation) 是指一个过程使用不同的安全上下文而不是拥有这个过程的一种 的能力。 20.10.29 Incident (事件 ) 一个事件 (Incident) 作为在一个信息系统或网络中的一个不利网络事件或这样的一个事 件威胁的发生。']
# embeddings_1 = model.encode(query)
# embeddings_2 = model.encode(contexts)
# similarity = embeddings_1 @ embeddings_2.T
# print(similarity)

