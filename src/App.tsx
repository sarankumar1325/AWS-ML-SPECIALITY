import { useState, useEffect } from 'react';
import { useQuiz } from './hooks/useQuiz';
import { StartScreen } from './components/quiz/StartScreen';
import { QuestionCard } from './components/quiz/QuestionCard';
import { FeedbackCard } from './components/quiz/FeedbackCard';
import { ResultsScreen } from './components/quiz/ResultsScreen';
import type { Question, QuizResults } from './types/quiz';
import { QuestionType } from './types/quiz';

const sampleQuestions: Question[] = [
  {
    id: '1',
    type: QuestionType.MCQ,
    question: 'A large mobile network operating company is building a machine learning model to predict customers who are likely to unsubscribe from the service. The company plans to offer an incentive for these customers as the cost of churn is far greater than the cost of the incentive. The model produces the following confusion matrix after evaluating on a test dataset of 100 customers. Based on the model evaluation results, why is this a viable model for production?',
    options: [
      { id: '1a', text: 'The model is 86% accurate and the cost incurred by the company as a result of false negatives is less than the false positives.', isCorrect: false },
      { id: '1b', text: 'The precision of the model is 86%, which is less than the accuracy of the model.', isCorrect: false },
      { id: '1c', text: 'The model is 86% accurate and the cost incurred by the company as a result of false positives is less than the false negatives.', isCorrect: true },
      { id: '1d', text: 'The precision of the model is 86%, which is greater than the accuracy of the model.', isCorrect: false },
    ],
    explanation: 'The model is 86% accurate. The key consideration is that false negatives (missing customers who will churn) cost more than false positives (offering incentives to customers who wouldn\'t have churned). Since the cost of churn is greater than the cost of incentives, having more false positives is acceptable. This makes the model viable for production.',
  },
  {
    id: '2',
    type: QuestionType.MCQ,
    question: 'A Machine Learning Specialist is designing a system for improving sales for a company. The objective is to use the large amount of information the company has on users\' behavior and product preferences to predict which products users would like based on the users\' similarity to other users. What should the Specialist do to meet this objective?',
    options: [
      { id: '2a', text: 'Build a content-based filtering recommendation engine with Apache Spark ML on Amazon EMR', isCorrect: false },
      { id: '2b', text: 'Build a collaborative filtering recommendation engine with Apache Spark ML on Amazon EMR', isCorrect: true },
      { id: '2c', text: 'Build a model-based filtering recommendation engine with Apache Spark ML on Amazon EMR', isCorrect: false },
      { id: '2d', text: 'Build a combinative filtering recommendation engine with Apache Spark ML on Amazon EMR', isCorrect: false },
    ],
    explanation: 'Collaborative filtering recommends items based on user similarity - if User A is similar to User B and User A liked an item, then User B would likely like that item too. This matches the objective of using user behavior and preferences to predict products based on similarity to other users.',
  },
  {
    id: '3',
    type: QuestionType.MCQ,
    question: 'A Mobile Network Operator is building an analytics platform to analyze and optimize company operations using Amazon Athena and Amazon S3. The source systems send data in .CSV format in real time. The Data Engineering team wants to transform the data to the Apache Parquet format before storing it on Amazon S3. Which solution takes the LEAST effort to implement?',
    options: [
      { id: '3a', text: 'Ingest .CSV data using Apache Kafka Streams on Amazon EC2 instances and use Kafka Connect S3 to serialize data as Parquet', isCorrect: false },
      { id: '3b', text: 'Ingest .CSV data from Amazon Kinesis Data Streams and use Amazon Glue to convert data into Parquet.', isCorrect: false },
      { id: '3c', text: 'Ingest .CSV data using Apache Spark Structured Streaming in an Amazon EMR cluster and use Apache Spark to convert data into Parquet.', isCorrect: false },
      { id: '3d', text: 'Ingest .CSV data from Amazon Kinesis Data Streams and use Amazon Kinesis Data Firehose to convert data into Parquet.', isCorrect: true },
    ],
    explanation: 'Amazon Kinesis Data Firehose provides a fully managed solution for ingesting streaming data and can automatically convert data to Parquet format with minimal configuration. It requires the least effort compared to setting up custom infrastructure with Kafka, EMR clusters, or Glue jobs.',
  },
  {
    id: '4',
    type: QuestionType.MCQ,
    question: 'A city wants to monitor its air quality to address the consequences of air pollution. A Machine Learning Specialist needs to forecast the air quality in parts per million of contaminants for the next 2 days in the city. As this is a prototype, only daily data from the last year is available. Which model is MOST likely to provide the best results in Amazon SageMaker?',
    options: [
      { id: '4a', text: 'Use the Amazon SageMaker k-Nearest-Neighbors (kNN) algorithm on the single time series consisting of the full year of data with a predictor_type of regressor.', isCorrect: false },
      { id: '4b', text: 'Use Amazon SageMaker Random Cut Forest (RCF) on the single time series consisting of the full year of data.', isCorrect: false },
      { id: '4c', text: 'Use the Amazon SageMaker Linear Learner algorithm on the single time series consisting of the full year of data with a predictor_type of regressor.', isCorrect: true },
      { id: '4d', text: 'Use the Amazon SageMaker Linear Learner algorithm on the single time series consisting of the full year of data with a predictor_type of classifier.', isCorrect: false },
    ],
    explanation: 'This is a regression problem (predicting a continuous value - parts per million). The Linear Learner algorithm with predictor_type set to "regressor" is appropriate for time series forecasting when you have a single time series and want to predict future values.',
  },
  {
    id: '5',
    type: QuestionType.MCQ,
    question: 'A Data Engineer needs to build a model using a dataset containing customer credit card information. How can the Data Engineer ensure the data remains encrypted and the credit card information is secure?',
    options: [
      { id: '5a', text: 'Use a custom encryption algorithm to encrypt the data and store the data on an Amazon SageMaker instance in a VPC. Use the SageMaker DeepAR algorithm to randomize the credit card numbers.', isCorrect: false },
      { id: '5b', text: 'Use an IAM policy to encrypt the data on the Amazon S3 bucket and Amazon Kinesis to automatically discard credit card numbers and insert fake credit card numbers.', isCorrect: false },
      { id: '5c', text: 'Use an Amazon SageMaker launch configuration to encrypt the data once it is copied to the SageMaker instance in a VPC. Use the SageMaker principal component analysis (PCA) algorithm to reduce the length of the credit card numbers.', isCorrect: false },
      { id: '5d', text: 'Use AWS KMS to encrypt the data on Amazon S3 and Amazon SageMaker, and redact the credit card numbers from the customer data with AWS Glue.', isCorrect: true },
    ],
    explanation: 'AWS KMS provides server-side encryption for data at rest in S3 and SageMaker. AWS Glue can be used to transform and redact sensitive data (like credit card numbers) from the dataset. This combination ensures data security and compliance with PCI DSS requirements.',
  },
  {
    id: '6',
    type: QuestionType.MCQ,
    question: 'A Machine Learning Specialist is using an Amazon SageMaker notebook instance in a private subnet of a corporate VPC. The ML Specialist has important data stored on the Amazon SageMaker notebook instance\'s Amazon EBS volume, and needs to take a snapshot of that EBS volume. However, the ML Specialist cannot find the Amazon SageMaker notebook instance\'s EBS volume or Amazon EC2 instance within the VPC. Why is the ML Specialist not seeing the instance visible in the VPC?',
    options: [
      { id: '6a', text: 'Amazon SageMaker notebook instances are based on the EC2 instances within the customer account, but they run outside of VPCs.', isCorrect: false },
      { id: '6b', text: 'Amazon SageMaker notebook instances are based on the Amazon ECS service within customer accounts.', isCorrect: false },
      { id: '6c', text: 'Amazon SageMaker notebook instances are based on EC2 instances running within AWS service accounts.', isCorrect: true },
      { id: '6d', text: 'Amazon SageMaker notebook instances are based on AWS ECS instances running within AWS service accounts.', isCorrect: false },
    ],
    explanation: 'Amazon SageMaker notebook instances are based on EC2 instances running in AWS service accounts, not in the customer\'s VPC. This is why the ML Specialist cannot find the instance or its EBS volume within the VPC. The notebook instances run in AWS-managed infrastructure.',
  },
  {
    id: '7',
    type: QuestionType.MCQ,
    question: 'A Machine Learning Specialist is building a model that will perform time series forecasting using Amazon SageMaker. The Specialist has finished training the model and is now planning to perform load testing on the endpoint so they can configure Auto Scaling for the model variant. Which approach will allow the Specialist to review the latency, memory utilization, and CPU utilization during the load test?',
    options: [
      { id: '7a', text: 'Review SageMaker logs that have been written to Amazon S3 by leveraging Amazon Athena and Amazon QuickSight to visualize logs as they are being produced.', isCorrect: false },
      { id: '7b', text: 'Generate an Amazon CloudWatch dashboard to create a single view for the latency, memory utilization, and CPU utilization metrics that are outputted by Amazon SageMaker.', isCorrect: true },
      { id: '7c', text: 'Build custom Amazon CloudWatch Logs and then leverage Amazon ES and Kibana to query and visualize the log data as it is generated by Amazon SageMaker.', isCorrect: false },
      { id: '7d', text: 'Send Amazon CloudWatch Logs that were generated by Amazon SageMaker to Amazon ES and use Kibana to query and visualize the log data.', isCorrect: false },
    ],
    explanation: 'Amazon CloudWatch automatically collects metrics from SageMaker endpoints including latency, memory utilization, and CPU utilization. Creating a CloudWatch dashboard provides a unified view of all these metrics during load testing, making it the easiest approach to configure Auto Scaling.',
  },
  {
    id: '8',
    type: QuestionType.MCQ,
    question: 'A manufacturing company has structured and unstructured data stored in an Amazon S3 bucket. A Machine Learning Specialist wants to use SQL to run queries on this data. Which solution requires the LEAST effort to be able to query this data?',
    options: [
      { id: '8a', text: 'Use AWS Data Pipeline to transform the data and Amazon RDS to run queries.', isCorrect: false },
      { id: '8b', text: 'Use AWS Glue to catalogue the data and Amazon Athena to run queries.', isCorrect: true },
      { id: '8c', text: 'Use AWS Batch to run ETL on the data and Amazon Aurora to run the queries.', isCorrect: false },
      { id: '8d', text: 'Use AWS Lambda to transform the data and Amazon Kinesis Data Analytics to run queries.', isCorrect: false },
    ],
    explanation: 'AWS Glue can automatically crawl S3 data and create a catalog, and Amazon Athena can directly query data in S3 using SQL without needing to load or transform it first. This serverless approach requires the least effort compared to building ETL pipelines.',
  },
  {
    id: '9',
    type: QuestionType.MCQ,
    question: 'A Machine Learning Specialist is developing a custom video recommendation model for an application. The dataset used to train this model is very large with millions of data points and is hosted in an Amazon S3 bucket. The Specialist wants to avoid loading all of this data onto an Amazon SageMaker notebook instance because it would take hours to move and will exceed the attached 5 GB Amazon EBS volume on the notebook instance. Which approach allows the Specialist to use all the data to train the model?',
    options: [
      { id: '9a', text: 'Load a smaller subset of the data into the SageMaker notebook and train locally. Confirm that the training code is executing and the model parameters seem reasonable. Initiate a SageMaker training job using the full dataset from the S3 bucket using Pipe input mode.', isCorrect: true },
      { id: '9b', text: 'Launch an Amazon EC2 instance with an AWS Deep Learning AMI and attach the S3 bucket to the instance. Train on a small amount of the data to verify the training code and hyperparameters. Go back to Amazon SageMaker and train using the full dataset.', isCorrect: false },
      { id: '9c', text: 'Use AWS Glue to train a model using a small subset of the data to confirm that the data will be compatible with Amazon SageMaker. Initiate a SageMaker training job using the full dataset from the S3 bucket using Pipe input mode.', isCorrect: false },
      { id: '9d', text: 'Load a smaller subset of the data into the SageMaker notebook and train locally. Confirm that the training code is executing and the model parameters seem reasonable. Launch an Amazon EC2 instance with an AWS Deep Learning AMI and attach the S3 bucket to train the full dataset.', isCorrect: false },
    ],
    explanation: 'Pipe input mode in SageMaker allows streaming data directly from S3 during training without needing to download it first. This avoids EBS volume limitations and allows training on the full dataset. The specialist can first validate with a smaller subset locally, then use Pipe mode for the full training job.',
  },
  {
    id: '10',
    type: QuestionType.MCQ,
    question: 'A Machine Learning Specialist has completed a proof of concept for a company using a small data sample, and now the Specialist is ready to implement an end-to-end solution in AWS using Amazon SageMaker. The historical training data is stored in Amazon RDS. Which approach should the Specialist use for training a model using that data?',
    options: [
      { id: '10a', text: 'Write a direct connection to the SQL database within the notebook and pull data in.', isCorrect: false },
      { id: '10b', text: 'Push the data from Microsoft SQL Server to Amazon S3 using an AWS Data Pipeline and provide the S3 location within the notebook.', isCorrect: true },
      { id: '10c', text: 'Move the data to Amazon DynamoDB and set up a connection to DynamoDB within the notebook to pull data in.', isCorrect: false },
      { id: '10d', text: 'Move the data to Amazon ElastiCache using AWS DMS and set up a connection within the notebook to pull data in for fast access.', isCorrect: false },
    ],
    explanation: 'AWS Data Pipeline can efficiently export data from Amazon RDS to Amazon S3. Once the data is in S3, SageMaker training jobs can easily access it. This approach is more scalable and secure than maintaining direct database connections from notebooks.',
  },
  {
    id: '11',
    type: QuestionType.MCQ,
    question: 'A Machine Learning Specialist receives customer data for an online shopping website. The data includes demographics, past visits, and locality information. The Specialist must develop a machine learning approach to identify the customer shopping patterns, preferences, and trends to enhance the website for better service and smart recommendations. Which solution should the Specialist recommend?',
    options: [
      { id: '11a', text: 'Latent Dirichlet Allocation (LDA) for the given collection of discrete data to identify patterns in the customer database.', isCorrect: false },
      { id: '11b', text: 'A neural network with a minimum of three layers and random initial weights to identify patterns in the customer database.', isCorrect: false },
      { id: '11c', text: 'Collaborative filtering based on user interactions and correlations to identify patterns in the customer database.', isCorrect: true },
      { id: '11d', text: 'Random Cut Forest (RCF) over random subsamples to identify patterns in the customer database.', isCorrect: false },
    ],
    explanation: 'Collaborative filtering is ideal for identifying shopping patterns, preferences, and trends based on user interactions and correlations. It recommends products by finding similar users or items, making it perfect for e-commerce recommendation systems.',
  },
  {
    id: '12',
    type: QuestionType.MCQ,
    question: 'A Machine Learning Specialist is working with a large company to leverage machine learning within its products. The company wants to group its customers into categories based on which customers will and will not churn within the next 6 months. The company has labeled the data available to the Specialist. Which machine learning model type should the Specialist use to accomplish this task?',
    options: [
      { id: '12a', text: 'Linear regression', isCorrect: false },
      { id: '12b', text: 'Classification', isCorrect: true },
      { id: '12c', text: 'Clustering', isCorrect: false },
      { id: '12d', text: 'Reinforcement learning', isCorrect: false },
    ],
    explanation: 'Since the data is labeled (churn/no churn) and the task is to categorize customers into discrete classes (will churn or will not churn), this is a classification problem. Classification algorithms predict categorical labels.',
  },
  {
    id: '13',
    type: QuestionType.MCQ,
    question: 'The displayed graph is from a forecasting model for testing a time series. Considering the graph only, which conclusion should a Machine Learning Specialist make about the behavior of the model?',
    options: [
      { id: '13a', text: 'The model predicts both the trend and the seasonality well', isCorrect: true },
      { id: '13b', text: 'The model predicts the trend well, but not the seasonality.', isCorrect: false },
      { id: '13c', text: 'The model predicts the seasonality well, but not the trend.', isCorrect: false },
      { id: '13d', text: 'The model does not predict the trend or the seasonality well.', isCorrect: false },
    ],
    explanation: 'When both the trend direction and seasonal patterns in the forecast closely match the actual data points on the graph, the model is successfully capturing both components. This indicates good model performance for time series forecasting.',
  },
  {
    id: '14',
    type: QuestionType.MCQ,
    question: 'A company wants to classify user behavior as either fraudulent or normal. Based on internal research, a Machine Learning Specialist would like to build a binary classifier based on two features: age of account and transaction month. The class distribution for these features is illustrated in the figure provided. Based on this information, which model would have the HIGHEST accuracy?',
    options: [
      { id: '14a', text: 'Long short-term memory (LSTM) model with scaled exponential linear unit (SELU)', isCorrect: false },
      { id: '14b', text: 'Logistic regression', isCorrect: false },
      { id: '14c', text: 'Support vector machine (SVM) with non-linear kernel', isCorrect: true },
      { id: '14d', text: 'Single perceptron with tanh activation function', isCorrect: false },
    ],
    explanation: 'When the class distribution shows non-linear separation between classes (fraudulent vs normal based on account age and transaction month), an SVM with a non-linear kernel can effectively find the optimal decision boundary. Simple linear models like logistic regression or single perceptrons would struggle with non-linear patterns.',
  },
  {
    id: '15',
    type: QuestionType.MCQ,
    question: 'A Machine Learning Specialist at a company sensitive to security is preparing a dataset for model training. The dataset is stored in Amazon S3 and contains Personally Identifiable Information (PII). The dataset: Must be accessible from a VPC only. Must not traverse the public internet. How can these requirements be satisfied?',
    options: [
      { id: '15a', text: 'Create a VPC endpoint and apply a bucket access policy that restricts access to the given VPC endpoint and the VPC.', isCorrect: true },
      { id: '15b', text: 'Create a VPC endpoint and apply a bucket access policy that allows access from the given VPC endpoint and an Amazon EC2 instance.', isCorrect: false },
      { id: '15c', text: 'Create a VPC endpoint and use Network Access Control Lists (NACLs) to allow traffic between only the given VPC endpoint and an Amazon EC2 instance.', isCorrect: false },
      { id: '15d', text: 'Create a VPC endpoint and use security groups to restrict access to the given VPC endpoint and an Amazon EC2 instance.', isCorrect: false },
    ],
    explanation: 'Creating a VPC endpoint (Gateway Endpoint for S3) and applying a bucket policy that restricts access to only that VPC endpoint ensures the data stays within the VPC and doesn\'t traverse the public internet. This provides the required security for PII data.',
  },
  {
    id: '16',
    type: QuestionType.MCQ,
    question: 'During mini-batch training of a neural network for a classification problem, a Data Scientist notices that training accuracy oscillates. What is the MOST likely cause of this issue?',
    options: [
      { id: '16a', text: 'The class distribution in the dataset is imbalanced.', isCorrect: false },
      { id: '16b', text: 'Dataset shuffling is disabled.', isCorrect: false },
      { id: '16c', text: 'The batch size is too big.', isCorrect: false },
      { id: '16d', text: 'The learning rate is very high.', isCorrect: true },
    ],
    explanation: 'A very high learning rate causes the optimizer to take large steps during gradient descent, leading to oscillations in the loss and training accuracy. This prevents the model from converging smoothly. Reducing the learning rate or using learning rate scheduling can help stabilize training.',
  },
  {
    id: '17',
    type: QuestionType.MCQ,
    question: 'An employee found a video clip with audio on a company\'s social media feed. The language used in the video is Spanish. English is the employee\'s first language, and they do not understand Spanish. The employee wants to do a sentiment analysis. What combination of services is the MOST efficient to accomplish the task?',
    options: [
      { id: '17a', text: 'Amazon Transcribe, Amazon Translate, and Amazon Comprehend', isCorrect: true },
      { id: '17b', text: 'Amazon Transcribe, Amazon Comprehend, and Amazon SageMaker seq2seq', isCorrect: false },
      { id: '17c', text: 'Amazon Transcribe, Amazon Translate, and Amazon SageMaker Neural Topic Model (NTM)', isCorrect: false },
      { id: '17d', text: 'Amazon Transcribe, Amazon Translate and Amazon SageMaker BlazingText', isCorrect: false },
    ],
    explanation: 'Amazon Transcribe converts Spanish audio to text, Amazon Translate converts Spanish text to English, and Amazon Comprehend performs sentiment analysis on the English text. This is the most efficient serverless combination without needing custom ML models.',
  },
  {
    id: '18',
    type: QuestionType.MCQ,
    question: 'A Machine Learning Specialist is packaging a custom ResNet model into a Docker container so the company can leverage Amazon SageMaker for training. The Specialist is using Amazon EC2 P3 instances to train the model and needs to properly configure the Docker container to leverage the NVIDIA GPUs. What does the Specialist need to do?',
    options: [
      { id: '18a', text: 'Bundle the NVIDIA drivers with the Docker image.', isCorrect: false },
      { id: '18b', text: 'Build the Docker container to be NVIDIA-Docker compatible.', isCorrect: true },
      { id: '18c', text: 'Organize the Docker container\'s file structure to execute on GPU instances.', isCorrect: false },
      { id: '18d', text: 'Set the GPU flag in the Amazon SageMaker CreateTrainingJob request body.', isCorrect: false },
    ],
    explanation: 'NVIDIA-Docker (nvidia-docker) allows Docker containers to access NVIDIA GPUs by providing the necessary runtime configuration. SageMaker training images are already NVIDIA-Docker compatible, and using this framework ensures proper GPU access for deep learning frameworks like TensorFlow or PyTorch.',
  },
  {
    id: '19',
    type: QuestionType.MCQ,
    question: 'A Machine Learning Specialist is building a logistic regression model that will predict whether or not a person will order a pizza. The Specialist is trying to build the optimal model with an ideal classification threshold. What model evaluation technique should the Specialist use to understand how different classification thresholds will impact the model\'s performance?',
    options: [
      { id: '19a', text: 'Receiver operating characteristic (ROC) curve', isCorrect: true },
      { id: '19b', text: 'Misclassification rate', isCorrect: false },
      { id: '19c', text: 'Root Mean Square Error (RMSE)', isCorrect: false },
      { id: '19d', text: 'L1 norm', isCorrect: false },
    ],
    explanation: 'The ROC curve plots True Positive Rate against False Positive Rate at various classification thresholds. It helps visualize how changing the threshold affects the trade-off between sensitivity and specificity, allowing the Specialist to select an optimal threshold based on business requirements.',
  },
  {
    id: '20',
    type: QuestionType.MCQ,
    question: 'An interactive online dictionary wants to add a widget that displays words used in similar contexts. A Machine Learning Specialist is asked to provide word features for the downstream nearest neighbor model powering the widget. What should the Specialist do to meet these requirements?',
    options: [
      { id: '20a', text: 'Create one-hot word encoding vectors.', isCorrect: false },
      { id: '20b', text: 'Produce a set of synonyms for every word using Amazon Mechanical Turk.', isCorrect: false },
      { id: '20c', text: 'Create word embedding vectors that store edit distance with every other word.', isCorrect: false },
      { id: '20d', text: 'Download word embeddings pre-trained on a large corpus.', isCorrect: true },
    ],
    explanation: 'Pre-trained word embeddings (like Word2Vec, GloVe, or FastText) capture semantic meaning and contextual similarity between words based on their usage in large text corpora. Words used in similar contexts will have similar embedding vectors, making them ideal for nearest neighbor queries in a dictionary widget.',
  },
  {
    id: '21',
    type: QuestionType.MSQ,
    question: 'A Machine Learning Specialist is configuring Amazon SageMaker so multiple Data Scientists can access notebooks, train models, and deploy endpoints. To ensure the best operational performance, the Specialist needs to be able to track how often the Scientists are deploying models, GPU and CPU utilization on the deployed SageMaker endpoints, and all errors that are generated when an endpoint is invoked. Which services are integrated with Amazon SageMaker to track this information? (Choose two.)',
    options: [
      { id: '21a', text: 'AWS CloudTrail', isCorrect: true },
      { id: '21b', text: 'AWS Health', isCorrect: false },
      { id: '21c', text: 'AWS Trusted Advisor', isCorrect: false },
      { id: '21d', text: 'Amazon CloudWatch', isCorrect: true },
      { id: '21e', text: 'AWS Config', isCorrect: false },
    ],
    explanation: 'AWS CloudTrail logs API calls including model deployments, and Amazon CloudWatch monitors metrics like GPU/CPU utilization and endpoint errors. These two services integrate with SageMaker to provide comprehensive operational visibility.',
  },
  {
    id: '22',
    type: QuestionType.MCQ,
    question: 'A retail chain has been ingesting purchasing records from its network of 20,000 stores to Amazon S3 using Amazon Kinesis Data Firehose. To support training an improved machine learning model, training records will require new but simple transformations, and some attributes will be combined. The model needs to be retrained daily. Given the large number of stores and the legacy data ingestion, which change will require the LEAST amount of development effort?',
    options: [
      { id: '22a', text: 'Require that the stores to switch to capturing their data locally on AWS Storage Gateway for loading into Amazon S3, then use AWS Glue to do the transformation.', isCorrect: false },
      { id: '22b', text: 'Deploy an Amazon EMR cluster running Apache Spark with the transformation logic, and have the cluster run each day on the accumulating records in Amazon S3, outputting new/transformed records to Amazon S3.', isCorrect: false },
      { id: '22c', text: 'Spin up a fleet of Amazon EC2 instances with the transformation logic, have them transform the data records accumulating on Amazon S3, and output the transformed records to Amazon S3.', isCorrect: false },
      { id: '22d', text: 'Insert an Amazon Kinesis Data Analytics stream downstream of the Kinesis Data Firehose stream that transforms raw record attributes into simple transformed values using SQL.', isCorrect: true },
    ],
    explanation: 'Kinesis Data Analytics allows SQL-based transformations on streaming data with minimal code. Adding it downstream of Kinesis Data Firehose requires the least effort compared to setting up EMR, EC2 fleets, or changing store-side data capture.',
  },
  {
    id: '23',
    type: QuestionType.MCQ,
    question: 'A Machine Learning Specialist is building a convolutional neural network (CNN) that will classify 10 types of animals. The Specialist has built a series of layers in a neural network that will take an input image of an animal, pass it through a series of convolutional and pooling layers, and then finally pass it through a dense and fully connected layer with 10 nodes. The Specialist would like to get an output from the neural network that is a probability distribution of how likely it is that the input image belongs to each of the 10 classes. Which function will produce the desired output?',
    options: [
      { id: '23a', text: 'Dropout', isCorrect: false },
      { id: '23b', text: 'Smooth L1 loss', isCorrect: false },
      { id: '23c', text: 'Softmax', isCorrect: true },
      { id: '23d', text: 'Rectified linear units (ReLU)', isCorrect: false },
    ],
    explanation: 'Softmax activation function converts the 10 output values into a probability distribution that sums to 1, where each value represents the probability of the input belonging to that class. This is the standard approach for multi-class classification output layers.',
  },
  {
    id: '24',
    type: QuestionType.MCQ,
    question: 'A Machine Learning Specialist trained a regression model, but the first iteration needs optimizing. The Specialist needs to understand whether the model is more frequently overestimating or underestimating the target. What option can the Specialist use to determine whether it is overestimating or underestimating the target value?',
    options: [
      { id: '24a', text: 'Root Mean Square Error (RMSE)', isCorrect: false },
      { id: '24b', text: 'Residual plots', isCorrect: true },
      { id: '24c', text: 'Area under the curve', isCorrect: false },
      { id: '24d', text: 'Confusion matrix', isCorrect: false },
    ],
    explanation: 'Residual plots (scatter plot of residuals vs predicted values) show whether predictions systematically overestimate (positive residuals) or underestimate (negative residuals) the target. A pattern in the residuals indicates bias in the model predictions.',
  },
  {
    id: '25',
    type: QuestionType.MCQ,
    question: 'A company wants to classify user behavior as either fraudulent or normal. Based on internal research, a Machine Learning Specialist would like to build a binary classifier based on two features: age of account and transaction month. The class distribution for these features is illustrated in the figure provided. Based on this information, which model would have the HIGHEST recall with respect to the fraudulent class?',
    options: [
      { id: '25a', text: 'Decision tree', isCorrect: true },
      { id: '25b', text: 'Linear support vector machine (SVM)', isCorrect: false },
      { id: '25c', text: 'Naive Bayesian classifier', isCorrect: false },
      { id: '25d', text: 'Single Perceptron with sigmoidal activation function', isCorrect: false },
    ],
    explanation: 'Decision trees can capture non-linear decision boundaries and are effective at identifying positive cases (fraudulent transactions). They tend to have higher recall because they can create rules that flag suspicious patterns without being overly constrained by linear separations.',
  },
  {
    id: '26',
    type: QuestionType.MCQ,
    question: 'A Machine Learning Specialist kicks off a hyperparameter tuning job for a tree-based ensemble model using Amazon SageMaker with Area Under the ROC Curve (AUC) as the objective metric. This workflow will eventually be deployed in a pipeline that retrains and tunes hyperparameters each night to model click-through on data that goes stale every 24 hours. With the goal of decreasing the amount of time it takes to train these models, and ultimately to decrease costs, the Specialist wants to reconfigure the input hyperparameter range(s). Which visualization will accomplish this?',
    options: [
      { id: '26a', text: 'A histogram showing whether the most important input feature is Gaussian.', isCorrect: false },
      { id: '26b', text: 'A scatter plot with points colored by target variable that uses t-Distributed Stochastic Neighbor Embedding (t-SNE) to visualize the large number of input variables in an easier-to-read dimension.', isCorrect: false },
      { id: '26c', text: 'A scatter plot showing the performance of the objective metric over each training iteration.', isCorrect: false },
      { id: '26d', text: 'A scatter plot showing the correlation between maximum tree depth and the objective metric.', isCorrect: true },
    ],
    explanation: 'A scatter plot showing the correlation between maximum tree depth and AUC helps identify the optimal hyperparameter range. If deeper trees consistently improve AUC, the range can be adjusted to focus on deeper values, reducing the search space and training time.',
  },
  {
    id: '27',
    type: QuestionType.MSQ,
    question: 'A Machine Learning Specialist is creating a new natural language processing application that processes a dataset comprised of 1 million sentences. The aim is to then run Word2Vec to generate embeddings of the sentences and enable different types of predictions. Here is an example from the dataset: "The quck BROWN FOX jumps over the lazy dog." Which of the following are the operations the Specialist needs to perform to correctly sanitize and prepare the data in a repeatable manner? (Choose three.)',
    options: [
      { id: '27a', text: 'Perform part-of-speech tagging and keep the action verb and the nouns only.', isCorrect: false },
      { id: '27b', text: 'Normalize all words by making the sentence lowercase.', isCorrect: true },
      { id: '27c', text: 'Remove stop words using an English stopword dictionary.', isCorrect: true },
      { id: '27d', text: 'Correct the typography on "quck" to "quick".', isCorrect: false },
      { id: '27e', text: 'One-hot encode all words in the sentence.', isCorrect: false },
      { id: '27f', text: 'Tokenize the sentence into words.', isCorrect: true },
    ],
    explanation: 'Word2Vec requires normalized text (lowercase), removal of common stop words, and tokenization into words. These preprocessing steps ensure consistent input for training embeddings. Typo correction and one-hot encoding are not required for Word2Vec.',
  },
  {
    id: '28',
    type: QuestionType.MCQ,
    question: 'A company is using Amazon Polly to translate plaintext documents to speech for automated company announcements. However, company acronyms are being mispronounced in the current documents. How should a Machine Learning Specialist address this issue for future documents?',
    options: [
      { id: '28a', text: 'Convert current documents to SSML with pronunciation tags.', isCorrect: false },
      { id: '28b', text: 'Create an appropriate pronunciation lexicon.', isCorrect: true },
      { id: '28c', text: 'Output speech marks to guide in pronunciation.', isCorrect: false },
      { id: '28d', text: 'Use Amazon Lex to preprocess the text files for pronunciation', isCorrect: false },
    ],
    explanation: 'Amazon Polly pronunciation lexicons allow you to define custom pronunciations for specific words or phrases, including company acronyms. This ensures correct pronunciation without modifying the source documents or using SSML tags.',
  },
  {
    id: '29',
    type: QuestionType.MSQ,
    question: 'An insurance company is developing a new device for vehicles that uses a camera to observe drivers\' behavior and alert them when they appear distracted. The company created approximately 10,000 training images in a controlled environment that a Machine Learning Specialist will use to train and evaluate machine learning models. During the model evaluation, the Specialist notices that the training error rate diminishes faster as the number of epochs increases and the model is not accurately inferring on the unseen test images. Which of the following should be used to resolve this issue? (Choose two.)',
    options: [
      { id: '29a', text: 'Add vanishing gradient to the model.', isCorrect: false },
      { id: '29b', text: 'Perform data augmentation on the training data.', isCorrect: true },
      { id: '29c', text: 'Make the neural network architecture complex.', isCorrect: false },
      { id: '29d', text: 'Use gradient checking in the model.', isCorrect: false },
      { id: '29e', text: 'Add L2 regularization to the model.', isCorrect: true },
    ],
    explanation: 'The symptoms indicate overfitting (low training error but poor test performance). Data augmentation increases training diversity to improve generalization, while L2 regularization penalizes large weights to reduce overfitting.',
  },
  {
    id: '30',
    type: QuestionType.MSQ,
    question: 'When submitting Amazon SageMaker training jobs using one of the built-in algorithms, which common parameters MUST be specified? (Choose three.)',
    options: [
      { id: '30a', text: 'The training channel identifying the location of training data on an Amazon S3 bucket.', isCorrect: true },
      { id: '30b', text: 'The validation channel identifying the location of validation data on an Amazon S3 bucket.', isCorrect: false },
      { id: '30c', text: 'The IAM role that Amazon SageMaker can assume to perform tasks on behalf of the users.', isCorrect: true },
      { id: '30d', text: 'Hyperparameters in a JSON array as documented for the algorithm used.', isCorrect: false },
      { id: '30e', text: 'The Amazon EC2 instance class specifying whether training will be run using CPU or GPU.', isCorrect: false },
      { id: '30f', text: 'The output path specifying where on an Amazon S3 bucket the trained model will persist.', isCorrect: true },
    ],
    explanation: 'SageMaker training jobs require: (1) training channel with S3 data location, (2) IAM role for AWS permissions, and (3) output path for model artifacts. Validation channel, hyperparameters, and instance type have defaults or are optional.',
  },
  {
    id: '31',
    type: QuestionType.MCQ,
    question: 'A monitoring service generates 1 TB of scale metrics record data every minute. A Research team performs queries on this data using Amazon Athena. The queries run slowly due to the large volume of data, and the team requires better performance. How should the records be stored in Amazon S3 to improve query performance?',
    options: [
      { id: '31a', text: 'CSV files', isCorrect: false },
      { id: '31b', text: 'Parquet files', isCorrect: true },
      { id: '31c', text: 'Compressed JSON', isCorrect: false },
      { id: '31d', text: 'RecordIO', isCorrect: false },
    ],
    explanation: 'Parquet is a columnar storage format that allows Athena to read only the columns needed for a query, significantly improving performance for large datasets. It also provides better compression and schema evolution compared to CSV, JSON, or RecordIO.',
  },
  {
    id: '32',
    type: QuestionType.MCQ,
    question: 'Machine Learning Specialist is working with a media company to perform classification on popular articles from the company\'s website. The company is using random forests to classify how popular an article will be before it is published. The Specialist wants to convert the Day_Of_Week column to binary values. What technique should be used to convert this column to binary values?',
    options: [
      { id: '32a', text: 'Binarization', isCorrect: false },
      { id: '32b', text: 'One-hot encoding', isCorrect: true },
      { id: '32c', text: 'Tokenization', isCorrect: false },
      { id: '32d', text: 'Normalization transformation', isCorrect: false },
    ],
    explanation: 'One-hot encoding converts categorical variables (like Day_Of_Week with values Monday, Tuesday, etc.) into binary columns. Each category becomes its own column with 1 or 0 values, allowing tree-based models like random forests to effectively use categorical features.',
  },
  {
    id: '33',
    type: QuestionType.MSQ,
    question: 'A gaming company has launched an online game where people can start playing for free, but they need to pay if they choose to use certain features. The company needs to build an automated system to predict whether or not a new user will become a paid user within 1 year. The training dataset consists of 1,000 positive samples (paid users) and 999,000 negative samples. Using this dataset for training, the Data Science team trained a random forest model that converged with over 99% accuracy on the training set. However, the prediction results on a test dataset were not satisfactory. Which of the following approaches should the Data Science team take to mitigate this issue? (Choose two.)',
    options: [
      { id: '33a', text: 'Add more deep trees to the random forest to enable the model to learn more features.', isCorrect: false },
      { id: '33b', text: 'Include a copy of the samples in the test dataset in the training dataset.', isCorrect: false },
      { id: '33c', text: 'Generate more positive samples by duplicating the positive samples and adding a small amount of noise to the duplicated data.', isCorrect: true },
      { id: '33d', text: 'Change the cost function so that false negatives have a higher impact on the cost value than false positives.', isCorrect: true },
      { id: '33e', text: 'Change the cost function so that false positives have a higher impact on the cost value than false negatives.', isCorrect: false },
    ],
    explanation: 'The dataset is highly imbalanced (1:999 ratio). SMOTE-like oversampling of positive samples addresses class imbalance. Adjusting the cost function to penalize false negatives more heavily helps the model prioritize detecting paying users, which is the rare positive class.',
  },
  {
    id: '34',
    type: QuestionType.MCQ,
    question: 'A Data Scientist is developing a machine learning model to predict future patient outcomes. The model should output a continuous value as its prediction. The data available includes labeled outcomes for a set of 4,000 patients. Out of 4,000 patient observations, there are 450 where the patient age has been input as 0. The other features for these observations appear normal compared to the rest of the sample population. How should the Data Scientist correct this issue?',
    options: [
      { id: '34a', text: 'Drop all records from the dataset where age has been set to 0.', isCorrect: false },
      { id: '34b', text: 'Replace the age field value for records with a value of 0 with the mean or median value from the dataset', isCorrect: true },
      { id: '34c', text: 'Drop the age feature from the dataset and train the model using the rest of the features.', isCorrect: false },
      { id: '34d', text: 'Use k-means clustering to handle missing features', isCorrect: false },
    ],
    explanation: 'Replacing invalid values (0) with the mean or median preserves the data and statistical distribution while keeping the valuable features from the other 4,500 valid observations. Dropping 450 records (11% of data) or removing the age feature would lose important information.',
  },
  {
    id: '35',
    type: QuestionType.MCQ,
    question: 'A Data Science team is designing a dataset repository where it will store a large amount of training data commonly used in its machine learning models. As Data Scientists may create an arbitrary number of new datasets every day, the solution has to scale automatically and be cost-effective. Also, it must be possible to explore the data using SQL. Which storage scheme is MOST adapted to this scenario?',
    options: [
      { id: '35a', text: 'Store datasets as files in Amazon S3.', isCorrect: true },
      { id: '35b', text: 'Store datasets as files in an Amazon EBS volume attached to an Amazon EC2 instance.', isCorrect: false },
      { id: '35c', text: 'Store datasets as tables in a multi-node Amazon Redshift cluster.', isCorrect: false },
      { id: '35d', text: 'Store datasets as global tables in Amazon DynamoDB.', isCorrect: false },
    ],
    explanation: 'Amazon S3 scales automatically, is cost-effective (pay only for usage), and supports SQL queries through Amazon Athena. It eliminates the need to manage infrastructure while providing durability and accessibility for ML training data.',
  },
];

function App() {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const [quizResults, setQuizResults] = useState<QuizResults | null>(null);

  useEffect(() => {
    const checkMedia = () => {
      setIsMobile(window.matchMedia('(max-width: 768px)').matches);
    };
    checkMedia();
    const query = window.matchMedia('(max-width: 768px)');
    query.addEventListener('change', checkMedia);
    return () => query.removeEventListener('change', checkMedia);
  }, []);

  const quiz = useQuiz({
    questions: sampleQuestions,
    onComplete: (results) => {
      setQuizResults(results);
    },
  });

  const handleStartQuiz = () => {
    quiz.startQuiz();
  };

  const handleSubmitAnswer = async () => {
    setIsSubmitting(true);
    await new Promise((resolve) => setTimeout(resolve, 300));
    quiz.submitAnswer();
    setIsSubmitting(false);
  };

  const handleNextQuestion = () => {
    quiz.nextQuestion();
  };

  const handleRestartQuiz = () => {
    quiz.restartQuiz();
  };

  const headerPadding = isMobile ? '12px 16px' : '12px 48px';

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#f8fafc' }}>
      <header style={{
        backgroundColor: 'white',
        borderBottom: '1px solid #e2e8f0',
        padding: headerPadding,
        height: isMobile ? '56px' : '65px',
        display: 'flex',
        alignItems: 'center',
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          maxWidth: '1200px',
          width: '100%',
          margin: '0 auto',
        }}>
          <div style={{
            width: isMobile ? '24px' : '28px',
            height: isMobile ? '24px' : '28px',
            backgroundColor: '#0f172a',
            borderRadius: '6px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}>
            <svg width={isMobile ? "14" : "16"} height={isMobile ? "14" : "16"} viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          </div>
          <span style={{ fontSize: isMobile ? '14px' : '16px', fontWeight: '600', color: '#0f172a' }}>
            AWS ML Specialty Quiz
          </span>
        </div>
      </header>

      {quiz.quizState === 'start' && (
        <StartScreen
          totalQuestions={quiz.totalQuestions}
          onStart={handleStartQuiz}
        />
      )}

      {quiz.quizState === 'question' && quiz.currentQuestion && (
        <QuestionCard
          question={quiz.currentQuestion}
          selectedOptionIds={quiz.currentAnswer?.selectedOptionIds || []}
          currentQuestionIndex={quiz.currentQuestionIndex + 1}
          totalQuestions={quiz.totalQuestions}
          showFeedback={false}
          onOptionToggle={quiz.toggleOption}
          onSubmit={handleSubmitAnswer}
          isSubmitting={isSubmitting}
        />
      )}

      {quiz.quizState === 'feedback' && quiz.currentQuestion && quiz.currentAnswer && (
        <FeedbackCard
          question={quiz.currentQuestion}
          userAnswer={quiz.currentAnswer}
          onNext={handleNextQuestion}
          isLastQuestion={quiz.currentQuestionIndex === quiz.totalQuestions - 1}
        />
      )}

      {quiz.quizState === 'results' && quizResults && (
        <ResultsScreen
          results={quizResults}
          onRestart={handleRestartQuiz}
        />
      )}
    </div>
  );
}

export default App;
