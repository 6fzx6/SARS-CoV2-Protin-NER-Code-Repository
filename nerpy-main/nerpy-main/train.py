# -*- coding: utf-8 -*-

import time
import random
import os
from tqdm import tqdm, trange
import logging
from loguru import logger

# 配置日志
logging.basicConfig(level=logging.INFO)
logger.add("simulated_model.log", rotation="500 MB")

class SimulatedNERModel:
    """
    """

    def __init__(self):
        self.model_name = "simulated-bert-ner"
        self.labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
        self.device = "cuda:0" if random.random() > 0.3 else "cpu"
        logger.info(f"初始化模型: {self.model_name}")
        logger.info(f"使用设备: {self.device}")
        time.sleep(1)

    def train_model(self, train_data, output_dir="outputs/", eval_data=None, **kwargs):
        """

        Args:
            train_data: 训练数据路径
            output_dir: 输出目录
            eval_data: 验证数据路径
        """
        logger.info("开始模型训练...")
        logger.info(f"训练数据: {train_data}")
        if eval_data:
            logger.info(f"验证数据: {eval_data}")
        logger.info(f"输出目录: {output_dir}")

        num_epochs = 300
        steps_per_epoch = 100
        total_steps = num_epochs * steps_per_epoch

        logger.info(f"训练轮数: {num_epochs}")
        logger.info(f"每轮步数: {steps_per_epoch}")
        logger.info(f"总步数: {total_steps}")

        for epoch in trange(num_epochs, desc="训练周期"):
            epoch_loss = 0.0
            epoch_iterator = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

            for step in epoch_iterator:
                time.sleep(0.05) 

                step_loss = random.uniform(0.1, 0.5)
                epoch_loss += step_loss

                epoch_iterator.set_postfix({"loss": f"{step_loss:.4f}"})

                if step > 0 and step % 20 == 0:
                    logger.info(f"  第 {step} 步 - 损失: {step_loss:.4f}")

                if step > 0 and step % 50 == 0:
                    logger.info(f"  保存检查点 checkpoint-{epoch+1}-{step}")
                    time.sleep(0.5) 

            avg_epoch_loss = epoch_loss / steps_per_epoch
            logger.info(f"第 {epoch+1} 轮完成 - 平均损失: {avg_epoch_loss:.4f}")

            if eval_data and (epoch+1) % 1 == 0:
                logger.info("执行验证...")
                eval_results = self._simulate_evaluation()
                logger.info(f"验证结果: {eval_results}")
                time.sleep(1)  
        # 保存最终模型
        logger.info("保存最终模型...")
        time.sleep(2)
        logger.info(f"模型已保存到 {output_dir}")
        logger.info("训练完成!")

        return total_steps, {"final_loss": avg_epoch_loss}

    def _simulate_evaluation(self):
        """

        Returns:
        """
        time.sleep(1)  

        eval_loss = random.uniform(0.1, 0.3)
        precision = random.uniform(0.8, 0.95)
        recall = random.uniform(0.75, 0.9)
        f1_score = random.uniform(0.8, 0.92)

        return {
            "eval_loss": round(eval_loss, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4)
        }

    def eval_model(self, eval_data, output_dir="eval_output/", **kwargs):
        """

        Args:
            eval_data: 评估数据路径
            output_dir: 输出目录

        Returns:
            tuple: 评估结果、模型输出、预测列表
        """
        logger.info("开始模型评估...")
        logger.info(f"评估数据: {eval_data}")
        logger.info(f"输出目录: {output_dir}")

        logger.info("加载评估数据...")
        time.sleep(1)

        eval_steps = 50
        eval_iterator = tqdm(range(eval_steps), desc="评估进度")
        for step in eval_iterator:
            time.sleep(0.05)  
            eval_iterator.set_postfix({"processed": f"{step+1}/{eval_steps}"})

        eval_results = self._simulate_evaluation()

        # 保存评估结果
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "eval_results.txt"), "w", encoding="utf-8") as f:
            for key, value in eval_results.items():
                f.write(f"{key} = {value}\n")

        logger.info("评估完成!")
        logger.info(f"评估结果: {eval_results}")

        return eval_results, [], []

    def predict(self, to_predict, split_on_space=True):
        """

        Args:
            to_predict: 待预测的文本列表
            split_on_space: 是否按空格分割

        Returns:
            tuple: 预测结果、模型输出、实体列表
        """
        logger.info("开始模型预测...")
        logger.info(f"待预测句子数: {len(to_predict)}")

        predictions = []
        model_outputs = []
        entities = []

        predict_iterator = tqdm(to_predict, desc="预测进度")
        for sentence in predict_iterator:
            time.sleep(0.2)  

            words = sentence.split() if split_on_space else list(sentence)
            pred_tags = []
            sentence_entities = []

            # 为每个词生成随机标签
            for word in words:
                tag = random.choice(self.labels)
                pred_tags.append({word: tag})

                # 如果是实体标签，添加到实体列表
                if tag.startswith("B-"):
                    entity_type = tag[2:]
                    sentence_entities.append((word, entity_type))

            predictions.append(pred_tags)
            model_outputs.append([{"logits": [random.random() for _ in self.labels]} for _ in words])
            entities.append(sentence_entities)

            # 更新进度条
            predict_iterator.set_postfix({"entities": len(sentence_entities)})

        logger.info("预测完成!")
        return predictions, model_outputs, entities

def main():
    print("=" * 60)
    print("           模型运行演示程序")
    print("=" * 60)

    model = SimulatedNERModel()

    train_data = r"C:\Users\梓辰\Desktop\新建文件夹\嬉笑夺来谪仙笔\nerpy-main\nerpy-main\project-8-at-2023-10-13-03-25-3697a1ef.json"
    eval_data = r"C:\Users\梓辰\Desktop\新建文件夹\嬉笑夺来谪仙笔\nerpy-main\nerpy-main\project-8-at-2023-10-13-03-25-3697a1ef.json"

    print("\n1. 开始模型训练...")
    print("-" * 40)
    global_step, training_details = model.train_model(
        train_data=train_data,
        output_dir="outputs/best_model/",
        eval_data=eval_data,
        num_train_epochs=300
    )

    print(f"\n训练完成! 总步数: {global_step}")
    print(f"训练详情: {training_details}")

    # 等待时间
    time.sleep(1)

    print("\n2. 开始模型评估...")
    print("-" * 40)
    eval_result, _, _ = model.eval_model(eval_data="data/test.txt")

    print(f"\n评估结果: {eval_result}")

    #  等待时间
    time.sleep(1)

    print("\n3. 开始 模型预测...")
    print("-" * 40)
    test_sentences = [
         "The ongoing outbreak of a new coronavirus (2019-nCoV, or severe acute respiratory syndrome coronavirus 2 [SARS-CoV-2]) has caused an epidemic of the acute respiratory syndrome known as coronavirus disease (COVID-19) in humans. SARS-CoV-2 rapidly spread to multiple regions of China and multiple other countries, posing a serious threat to public health. The spike (S) proteins of SARS-CoV-1 and SARS-CoV-2 may use the same host cellular receptor, angiotensin-converting enzyme 2 (ACE2), for entering host cells. The affinity between ACE2 and the SARS-CoV-2 S protein is much higher than that of ACE2 binding to the SARS-CoV S protein, explaining why SARS-CoV-2 seems to be more readily transmitted from human to human. Here, we report that ACE2 can be significantly upregulated after infection of various viruses, including SARS-CoV-1 and SARS-CoV-2, or by the stimulation with inflammatory cytokines such as interferons. We propose that SARS-CoV-2 may positively induce its cellular entry receptor, ACE2, to accelerate its replication and spread; high inflammatory cytokine levels increase ACE2 expression and act as high-risk factors for developing COVID-19, and the infection of other viruses may increase the risk of SARS-CoV-2 infection. Therefore, drugs targeting ACE2 may be developed for the future emerging infectious diseases caused by this cluster of coronaviruses. ",
         "Coronavirus disease 2019 (COVID-19) is a newly emerging human infectious disease caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2, previously called 2019-nCoV). Based on the rapid increase in the rate of human infection, the World Health Organization (WHO) has classified the COVID-19 outbreak as a pandemic. Because no specific drugs or vaccines for COVID-19 are yet available, early diagnosis and management are crucial for containing the outbreak. Here, we report a field-effect transistor (FET)-based biosensing device for detecting SARS-CoV-2 in clinical samples. The sensor was produced by coating graphene sheets of the FET with a specific antibody against SARS-CoV-2 spike protein. The performance of the sensor was determined using antigen protein, cultured virus, and nasopharyngeal swab specimens from COVID-19 patients. Our FET device could detect the SARSCoV-2 spike protein at concentrations of 1 fg\/mL in phosphate-buffered saline and 100 fg\/mL clinical transport medium. In addition, the FET sensor successfully detected SARS-CoV-2 in culture medium (limit of detection [LOD]: 1.6 X 10(1) pfu\/mL) and clinical samples (LOD: 2.42 X 10(2) copies\/mL). Thus, we have successfully fabricated a promising FET biosensor for SARS-CoV-2; our device is a highly sensitive immunological diagnostic method for COVID-19 that requires no sample pretreatment or labeling. ",

        "Assessment of commercial severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) immunoassays for their capacity to provide reliable information on sera neutralizing activity is an emerging need. We evaluated the performance of two commercially available lateral flow immunochromatographic assays (LFIC; Wondfo SARS-CoV-2 Antibody test and the INNOVITA 2019-nCoV Ab test) in comparison with a SARS-CoV-2 neutralization pseudotyped assay for coronavirus disease 2019 (COVID-19) diagnosis in hospitalized patients and investigate whether the intensity of the test band in LFIC associates with neutralizing antibody (NtAb) titers. Ninety sera were included from 51 patients with moderate to severe COVID-19. A green fluorescent protein (GFP) reporter-based pseudotyped neutralization assay (vesicular stomatitis virus coated with SARS-CoV-2 spike protein) was used. Test line intensity was scored using a 4-level scale (0 to 3+). The overall sensitivity of LFIC assays was 91.1% for the Wondfo SARS-CoV-2 Antibody test, 72.2% for the INNOVITA 2019-nCoV IgG, 85.6% for the INNOVITA 2019-nCoV IgM, and 92.2% for the NtAb assay. Sensitivity increased for all assays in sera collected beyond day 14 after symptoms onset (93.9%, 79.6%, 93.9%, and 93.9%, respectively). Reactivities equal to or more intense than the positive control line (>= 2+) in the Wondfo assay had a negative predictive value of 100% and a positive predictive value of 96.4% for high NtAb(50) titers (>= 1\/160). Our findings support the use of LFIC assays evaluated herein, particularly the Wondfo test, for COVID-19 diagnosis. We also find evidence that these rapid immunoassays can be used to predict high SARS-CoV-2-S NtAb(50) titers. ",

         "The article highlights an up-to-date progress in studies on structural and the remedial aspects of novel coronavirus 2019-nCoV, renamed as SARS-CoV-2, leading to the disease COVID-19, a pandemic. In general, all CoVs including SARS-CoV-2 are spherical positive single-stranded RNA viruses containing spike (S) protein, envelope (E) protein, nucleocapsid (N) protein, and membrane (M) protein, where S protein has a Receptor-binding Domain (RBD) that mediates the binding to host cell receptor, Angiotensin Converting Enzyme 2 (ACE2). The article details the repurposing of some drugs to be tried for COVID-19 and presents the status of vaccine development so far. Besides drugs and vaccines, the role of Convalescent Plasma (CP) therapy to treat COVID-19 is also discussed. ",
        "The World Health Organization (WHO) has issued a warning that, although the 2019 novel coronavirus (COVID-19) from Wuhan City (China), is not pandemic, it should be contained to prevent the global spread. The COVID-19 virus was known earlier as 2019-nCoV. As of 12 February 2020, WHO reported 45,171 cases and 1115 deaths related to COVID-19. COVID-19 is similar to Severe Acute Respiratory Syndrome coronavirus (SARS-CoV) virus in its pathogenicity, clinical spectrum, and epidemiology. Comparison of the genome sequences of COVID-19, SARS-CoV, and Middle East Respiratory Syndrome coronavirus (MERS-CoV) showed that COVID-19 has a better sequence identity with SARS-CoV compared to MERS CoV. However, the amino acid sequence of COVID-19 differs from other coronaviruses specifically in the regions of lab polyprotein and surface glycoprotein or S-protein. Although several animals have been speculated to be a reservoir for COVID-19, no animal reservoir has been already confirmed. COVID-19 causes COVID-19 disease that has similar symptoms as SARSCoV. Studies suggest that the human receptor for COVID-19 may be angiotensin-converting enzyme 2 (ACE2) receptor similar to that of SARS-CoV. The nucleocapsid (N) protein of COVID-19 has nearly 90% amino acid sequence identity with SARS-CoV. The N protein antibodies of SARS-CoV may cross react with COVID-19 but may not provide cross-immunity. In a similar fashion to SARS-CoV, the N protein of COVID-19 may play an important role in suppressing the RNA interference (RNAi) to overcome the host defense. This mini-review aims at investigating the most recent trend of COVID-19. ",
        "For the clinical application of semi-quantitative anti-SARS-CoV-2 antibody tests, the analytical performance and titer correlation of the plaque reduction neutralization test (PRNT) need to be investigated. We evaluated the analytical performance and PRNT titer-correlation of one surrogate virus neutralization test (sVNT) kit and three chemiluminescent assays. We measured the total antibodies for the receptor-binding domain (RBD) of the spike protein, total antibodies for the nucleocapsid protein (NP), and IgG antibodies for the RBD. All three chemiluminescent assays showed high analytical performance for the detection of SARS-CoV-2 infection, with a sensitivity >= 98% and specificity >= 99%; those of the sVNT were slightly lower. The representativeness of the neutralizing activity of PRNT ND50 >= 20 was comparable among the four immunoassays (Cohen's kappa approximate to 0.80). Quantitative titer correlation for high PRNT titers of ND50 >= 50, 200, and 1,000 was investigated with new cut-off values; the anti-RBD IgG antibody kit showed the best performance. It also showed the best linear correlation with PRNT titer in both the acute and convalescent phases (Pearson's R 0.81 and 0.72, respectively). Due to the slowly waning titer of anti-NP antibodies, the correlation with PRNT titer at the convalescent phase was poor. In conclusion, semi-quantitative immunoassay kits targeting the RBD showed neutralizing activity that was correlated by titer; measurement of anti-NP antibodies would be useful for determining past infections. ",
        "Purpose Antibody assays against SARS-CoV-2 are used in sero-epidemiological studies to estimate the proportion of a population with past infection. IgG antibodies against the spike protein (S-IgG) allow no distinction between infection and vaccination. We evaluated the role of anti-nucleocapsid-IgG (N-IgG) to identify individuals with infection more than one year past infection. Methods S- and N-IgG were determined using the Euroimmun enzyme-linked immunosorbent assay (ELISA) in two groups: a randomly selected sample from the population of Stuttgart, Germany, and individuals with PCR-proven SARS-CoV-2 infection. Participants were five years or older. Demographics and comorbidities were registered from participants above 17 years. Results Between June 15, 2021 and July 14, 2021, 454 individuals from the random sample participated, as well as 217 individuals with past SARS-CoV-2 infection. Mean time from positive PCR test result to antibody testing was 458.7 days (standard deviation 14.6 days) in the past infection group. In unvaccinated individuals, the seroconversion rate for S-IgG was 25.5% in the random sample and 75% in the past infection group (P = < 0.001). In vaccinated individuals, the mean signal ratios for S-IgG were higher in individuals with prior infection (6.9 vs 11.2; P = < 0.001). N-IgG were only detectable in 17.1% of participants with past infection. Predictors for detectable N-IgG were older age, male sex, fever, wheezing and in-hospital treatment for COVID-19 and cardiovascular comorbidities. Conclusion N-IgG is not a reliable marker for SARS-CoV-2 infection after more than one year. In future, other diagnostic tests are needed to identify individuals with past natural infection. ",

    ]

    predictions, model_outputs, entities = model.predict(test_sentences)

    print("\n预测结果:")
    for i, (sentence, ents) in enumerate(zip(test_sentences, entities)):
        print(f"  句子 {i+1}: {sentence}")
        if ents:
            print(f"    识别实体: {ents}")
        else:
            print("    未识别到实体")

    print("\n" + "=" * 60)
    print("            运行完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
