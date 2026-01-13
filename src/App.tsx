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
  {
    id: '36',
    type: QuestionType.MCQ,
    question: 'A Machine Learning Specialist deployed a model that provides product recommendations on a company\'s website. Initially, the model was performing very well. However, within the past few months, the effect of product recommendations has diminished as customers are returning to their original habits. The model has not changed from its initial deployment over a year ago. Which method should the Specialist try to improve model performance?',
    options: [
      { id: '36a', text: 'The model needs to be completely re-engineered because it is unable to handle product inventory changes.', isCorrect: false },
      { id: '36b', text: 'The model\'s hyperparameters should be periodically updated to prevent drift.', isCorrect: false },
      { id: '36c', text: 'The model should be periodically retrained from scratch using the original data while adding a regularization term to handle product inventory changes.', isCorrect: false },
      { id: '36d', text: 'The model should be periodically retrained using the original training data plus new data as product inventory changes.', isCorrect: true },
    ],
    explanation: 'Concept drift occurs when the underlying data distribution changes over time (e.g., product inventory, customer preferences). Periodically retraining the model with new data combined with historical data helps the model adapt to these changes and maintain recommendation quality.',
  },
  {
    id: '37',
    type: QuestionType.MCQ,
    question: 'A Machine Learning Specialist working for an online fashion company wants to build a data ingestion solution for the company\'s Amazon S3-based data lake. The Specialist wants to create mechanisms for: Real-time analytics, Interactive analytics of historical data, Clickstream analytics, Product recommendations. Which services should the Specialist use?',
    options: [
      { id: '37a', text: 'AWS Glue as the data catalog; Amazon Kinesis Data Streams and Amazon Kinesis Data Analytics for real-time data insights; Amazon Kinesis Data Firehose for delivery to Amazon ES for clickstream analytics; Amazon EMR to generate personalized product recommendations', isCorrect: true },
      { id: '37b', text: 'Amazon Athena as the data catalog: Amazon Kinesis Data Streams and Amazon Kinesis Data Analytics for near-real-time data insights; Amazon Kinesis Data Firehose for clickstream analytics; AWS Glue to generate personalized product recommendations', isCorrect: false },
      { id: '37c', text: 'AWS Glue as the data catalog; Amazon Kinesis Data Streams and Amazon Kinesis Data Analytics for historical data insights; Amazon Kinesis Data Firehose for delivery to Amazon ES for clickstream analytics; Amazon EMR to generate personalized product recommendations', isCorrect: false },
      { id: '37d', text: 'Amazon Athena as the data catalog; Amazon Kinesis Data Streams and Amazon Kinesis Data Analytics for historical data insights; Amazon DynamoDB streams for clickstream analytics; AWS Glue to generate personalized product recommendations', isCorrect: false },
    ],
    explanation: 'AWS Glue provides data cataloging, Kinesis Data Streams/Analytics enable real-time processing, Kinesis Data Firehose delivers to Amazon ES for clickstream analytics, and Amazon EMR generates product recommendations. This combination provides all required capabilities.',
  },
  {
    id: '38',
    type: QuestionType.MSQ,
    question: 'A company is observing low accuracy while training on the default built-in image classification algorithm in Amazon SageMaker. The Data Science team wants to use an Inception neural network architecture instead of a ResNet architecture. Which of the following will accomplish this? (Choose two.)',
    options: [
      { id: '38a', text: 'Customize the built-in image classification algorithm to use Inception and use this for model training.', isCorrect: false },
      { id: '38b', text: 'Create a support case with the SageMaker team to change the default image classification algorithm to Inception.', isCorrect: false },
      { id: '38c', text: 'Bundle a Docker container with TensorFlow Estimator loaded with an Inception network and use this for model training.', isCorrect: true },
      { id: '38d', text: 'Use custom code in Amazon SageMaker with TensorFlow Estimator to load the model with an Inception network, and use this for model training.', isCorrect: true },
      { id: '38e', text: 'Download and apt-get install the inception network code into an Amazon EC2 instance and use this instance as a Jupyter notebook in Amazon SageMaker.', isCorrect: false },
    ],
    explanation: 'To use Inception instead of the built-in ResNet algorithm, the team needs custom code with TensorFlow Estimator. This can be done by either bundling a custom Docker container or using custom code in SageMaker with TensorFlow Estimator loaded with the Inception architecture.',
  },
  {
    id: '39',
    type: QuestionType.MCQ,
    question: 'A Machine Learning Specialist built an image classification deep learning model. However, the Specialist ran into an overfitting problem in which the training and testing accuracies were 99% and 75%, respectively. How should the Specialist address this issue and what is the reason behind it?',
    options: [
      { id: '39a', text: 'The learning rate should be increased because the optimization process was trapped at a local minimum.', isCorrect: false },
      { id: '39b', text: 'The dropout rate at the flatten layer should be increased because the model is not generalized enough.', isCorrect: true },
      { id: '39c', text: 'The dimensionality of dense layer next to the flatten layer should be increased because the model is not complex enough.', isCorrect: false },
      { id: '39d', text: 'The epoch number should be increased because the optimization process was terminated before it reached the global minimum.', isCorrect: false },
    ],
    explanation: 'The large gap between training (99%) and testing (75%) accuracy indicates overfitting. Increasing dropout rate randomly deactivates neurons during training, forcing the network to learn more robust features and generalize better to unseen data.',
  },
  {
    id: '40',
    type: QuestionType.MCQ,
    question: 'A Machine Learning team uses Amazon SageMaker to train an Apache MXNet handwritten digit classifier model. The team wants to receive a notification when the model is overfitting. Auditors want to view the Amazon SageMaker log activity report to ensure there are no unauthorized API calls. What should the Machine Learning team do with the least amount of code and fewest steps?',
    options: [
      { id: '40a', text: 'Implement an AWS Lambda function to log Amazon SageMaker API calls to Amazon S3. Add code to push a custom metric to Amazon CloudWatch. Create an alarm in CloudWatch with Amazon SNS to receive a notification when the model is overfitting.', isCorrect: false },
      { id: '40b', text: 'Use AWS CloudTrail to log Amazon SageMaker API calls to Amazon S3. Add code to push a custom metric to Amazon CloudWatch. Create an alarm in CloudWatch with Amazon SNS to receive a notification when the model is overfitting.', isCorrect: true },
      { id: '40c', text: 'Implement an AWS Lambda function to log Amazon SageMaker API calls to AWS CloudTrail. Add code to push a custom metric to Amazon CloudWatch. Create an alarm in CloudWatch with Amazon SNS to receive a notification when the model is overfitting.', isCorrect: false },
      { id: '40d', text: 'Use AWS CloudTrail to log Amazon SageMaker API calls to Amazon S3. Set up Amazon SNS to receive a notification when the model is overfitting.', isCorrect: false },
    ],
    explanation: 'AWS CloudTrail automatically logs API calls with minimal configuration. Custom metrics pushed to CloudWatch combined with CloudWatch Alarms and SNS provide overfitting notifications. This approach requires the least code and steps compared to custom Lambda implementations.',
  },
  {
    id: '41',
    type: QuestionType.MCQ,
    question: 'A Machine Learning Specialist is building a prediction model for a large number of features using linear models, such as linear regression and logistic regression. During exploratory data analysis, the Specialist observes that many features are highly correlated with each other. This may make the model unstable. What should be done to reduce the impact of having such a large number of features?',
    options: [
      { id: '41a', text: 'Perform one-hot encoding on highly correlated features.', isCorrect: false },
      { id: '41b', text: 'Use matrix multiplication on highly correlated features.', isCorrect: false },
      { id: '41c', text: 'Create a new feature space using principal component analysis (PCA)', isCorrect: true },
      { id: '41d', text: 'Apply the Pearson correlation coefficient.', isCorrect: false },
    ],
    explanation: 'PCA transforms correlated features into a smaller set of uncorrelated principal components. This reduces multicollinearity in linear models, stabilizing the model and reducing variance in coefficient estimates.',
  },
  {
    id: '42',
    type: QuestionType.MCQ,
    question: 'A Machine Learning Specialist is implementing a full Bayesian network on a dataset that describes public transit in New York City. One of the random variables is discrete, and represents the number of minutes New Yorkers wait for a bus given that the buses cycle every 10 minutes, with a mean of 3 minutes. Which prior probability distribution should the ML Specialist use for this variable?',
    options: [
      { id: '42a', text: 'Poisson distribution', isCorrect: true },
      { id: '42b', text: 'Uniform distribution', isCorrect: false },
      { id: '42c', text: 'Normal distribution', isCorrect: false },
      { id: '42d', text: 'Binomial distribution', isCorrect: false },
    ],
    explanation: 'The Poisson distribution models the count of events occurring in a fixed interval (waiting time in minutes) when events happen independently at a constant average rate (mean of 3 minutes). It is appropriate for discrete, non-negative integer counts.',
  },
  {
    id: '43',
    type: QuestionType.MCQ,
    question: 'A Data Science team within a large company uses Amazon SageMaker notebooks to access data stored in Amazon S3 buckets. The IT Security team is concerned that internet-enabled notebook instances create a security vulnerability. The company mandates that all instances stay within a secured VPC with no internet access, and data communication traffic must stay within the AWS network. How should the Data Science team configure the notebook instance placement to meet these requirements?',
    options: [
      { id: '43a', text: 'Associate the Amazon SageMaker notebook with a private subnet in a VPC. Place the Amazon SageMaker endpoint and S3 buckets within the same VPC.', isCorrect: false },
      { id: '43b', text: 'Associate the Amazon SageMaker notebook with a private subnet in a VPC. Use IAM policies to grant access to Amazon S3 and Amazon SageMaker.', isCorrect: false },
      { id: '43c', text: 'Associate the Amazon SageMaker notebook with a private subnet in a VPC. Ensure the VPC has S3 VPC endpoints and Amazon SageMaker VPC endpoints attached to it.', isCorrect: true },
      { id: '43d', text: 'Associate the Amazon SageMaker notebook with a private subnet in a VPC. Ensure the VPC has a NAT gateway and an associated security group allowing only outbound connections to Amazon S3 and Amazon SageMaker.', isCorrect: false },
    ],
    explanation: 'VPC endpoints for S3 and SageMaker enable private connectivity within the AWS network without internet access. Gateway endpoints for S3 provide direct, secure access while VPC endpoints for SageMaker enable API calls within the VPC.',
  },
  {
    id: '44',
    type: QuestionType.MSQ,
    question: 'A Machine Learning Specialist has created a deep learning neural network model that performs well on the training data but performs poorly on the test data. Which of the following methods should the Specialist consider using to correct this? (Choose three.)',
    options: [
      { id: '44a', text: 'Decrease regularization.', isCorrect: false },
      { id: '44b', text: 'Increase regularization.', isCorrect: true },
      { id: '44c', text: 'Increase dropout.', isCorrect: true },
      { id: '44d', text: 'Decrease dropout.', isCorrect: false },
      { id: '44e', text: 'Increase feature combinations.', isCorrect: false },
      { id: '44f', text: 'Decrease feature combinations.', isCorrect: true },
    ],
    explanation: 'The model is overfitting (high training accuracy, low test accuracy). To reduce overfitting: increase regularization penalties, increase dropout rates to prevent co-adaptation of neurons, and decrease feature combinations to reduce model complexity.',
  },
  {
    id: '45',
    type: QuestionType.MCQ,
    question: 'A Data Scientist needs to create a serverless ingestion and analytics solution for high-velocity, real-time streaming data. The ingestion process must buffer and convert incoming records from JSON to a query-optimized, columnar format without data loss. The output datastore must be highly available, and Analysts must be able to run SQL queries against the data and connect to existing business intelligence dashboards. Which solution should the Data Scientist build to satisfy the requirements?',
    options: [
      { id: '45a', text: 'Create a schema in the AWS Glue Data Catalog of the incoming data format. Use an Amazon Kinesis Data Firehose delivery stream to stream the data and transform the data to Apache Parquet or ORC format using the AWS Glue Data Catalog before delivering to Amazon S3. Have the Analysts query the data directly from Amazon S3 using Amazon Athena, and connect to BI tools using the Athena Java Database Connectivity (JDBC) connector.', isCorrect: true },
      { id: '45b', text: 'Write each JSON record to a staging location in Amazon S3. Use the S3 Put event to trigger an AWS Lambda function that transforms the data into Apache Parquet or ORC format and writes the data to a processed data location in Amazon S3. Have the Analysts query the data directly from Amazon S3 using Amazon Athena, and connect to BI tools using the Athena Java Database Connectivity (JDBC) connector.', isCorrect: false },
      { id: '45c', text: 'Write each JSON record to a staging location in Amazon S3. Use the S3 Put event to trigger an AWS Lambda function that transforms the data into Apache Parquet or ORC format and inserts it into an Amazon RDS PostgreSQL database. Have the Analysts query and run dashboards from the RDS database.', isCorrect: false },
      { id: '45d', text: 'Use Amazon Kinesis Data Analytics to ingest the streaming data and perform real-time SQL queries to convert the records to Apache Parquet before delivering to Amazon S3. Have the Analysts query the data directly from Amazon S3 using Amazon Athena and connect to BI tools using the Athena Java Database Connectivity (JDBC) connector.', isCorrect: false },
    ],
    explanation: 'Kinesis Data Firehose with AWS Glue Data Catalog provides serverless ETL that automatically converts JSON to Parquet/ORC format. Amazon S3 ensures high availability, and Athena enables SQL queries with BI dashboard connectivity via JDBC.',
  },
  {
    id: '46',
    type: QuestionType.MCQ,
    question: 'An online reseller has a large, multi-column dataset with one column missing 30% of its data. A Machine Learning Specialist believes that certain columns in the dataset could be used to reconstruct the missing data. Which reconstruction approach should the Specialist use to preserve the integrity of the dataset?',
    options: [
      { id: '46a', text: 'Listwise deletion', isCorrect: false },
      { id: '46b', text: 'Last observation carried forward', isCorrect: false },
      { id: '46c', text: 'Multiple imputation', isCorrect: true },
      { id: '46d', text: 'Mean substitution', isCorrect: false },
    ],
    explanation: 'Multiple imputation creates multiple plausible estimates for missing values based on relationships in the data, preserving statistical properties and uncertainty. This is more sophisticated than simple methods like mean substitution or listwise deletion which can bias results.',
  },
  {
    id: '47',
    type: QuestionType.MCQ,
    question: 'A company is setting up an Amazon SageMaker environment. The corporate data security policy does not allow communication over the internet. How can the company enable the Amazon SageMaker service without enabling direct internet access to Amazon SageMaker notebook instances?',
    options: [
      { id: '47a', text: 'Create a NAT gateway within the corporate VPC.', isCorrect: false },
      { id: '47b', text: 'Route Amazon SageMaker traffic through an on-premises network.', isCorrect: false },
      { id: '47c', text: 'Create Amazon SageMaker VPC interface endpoints within the corporate VPC.', isCorrect: true },
      { id: '47d', text: 'Create VPC peering with Amazon VPC hosting Amazon SageMaker.', isCorrect: false },
    ],
    explanation: 'VPC interface endpoints (powered by AWS PrivateLink) enable private connectivity between the VPC and SageMaker without traversing the internet. This satisfies security requirements while allowing notebook instances to access SageMaker APIs.',
  },
  {
    id: '48',
    type: QuestionType.MCQ,
    question: 'A Machine Learning Specialist is training a model to identify the make and model of vehicles in images. The Specialist wants to use transfer learning and an existing model trained on images of general objects. The Specialist collated a large custom dataset of pictures containing different vehicle makes and models. What should the Specialist do to initialize the model to re-train it with the custom data?',
    options: [
      { id: '48a', text: 'Initialize the model with random weights in all layers including the last fully connected layer.', isCorrect: false },
      { id: '48b', text: 'Initialize the model with pre-trained weights in all layers and replace the last fully connected layer.', isCorrect: true },
      { id: '48c', text: 'Initialize the model with random weights in all layers and replace the last fully connected layer.', isCorrect: false },
      { id: '48d', text: 'Initialize the model with pre-trained weights in all layers including the last fully connected layer.', isCorrect: false },
    ],
    explanation: 'Transfer learning works best when initializing with pre-trained ImageNet weights for feature extraction layers while replacing the final classification layer with a new randomly initialized layer for vehicle make/model classification.',
  },
  {
    id: '49',
    type: QuestionType.MCQ,
    question: 'An office security agency conducted a successful pilot using 100 cameras. Images were uploaded to Amazon S3 and tagged using Amazon Rekognition. The agency wants to expand to thousands of cameras globally to identify activities performed by non-employees in real time. Which solution should the agency consider?',
    options: [
      { id: '49a', text: 'Use a proxy server at each local office and for each camera, and stream the RTSP feed to a unique Amazon Kinesis Video Streams video stream. On each stream, use Amazon Rekognition Video and create a stream processor to detect faces from a collection of known employees, and alert when non-employees are detected.', isCorrect: true },
      { id: '49b', text: 'Use a proxy server at each local office and for each camera, and stream the RTSP feed to a unique Amazon Kinesis Video Streams video stream. On each stream, use Amazon Rekognition Image to detect faces from a collection of known employees and alert when non-employees are detected.', isCorrect: false },
      { id: '49c', text: 'Install AWS DeepLens cameras and use the DeepLens_Kinesis_Video module to stream video to Amazon Kinesis Video Streams for each camera. On each stream, use Amazon Rekognition Video and create a stream processor to detect faces from a collection on each stream, and alert when non-employees are detected.', isCorrect: false },
      { id: '49d', text: 'Install AWS DeepLens cameras and use the DeepLens_Kinesis_Video module to stream video to Amazon Kinesis Video Streams for each camera. On each stream, run an AWS Lambda function to capture image fragments and then call Amazon Rekognition Image to detect faces from a collection of known employees, and alert when non-employees are detected.', isCorrect: false },
    ],
    explanation: 'Kinesis Video Streams with Rekognition Video stream processors enable real-time face detection and comparison against employee collections. This scales to thousands of cameras while providing real-time alerts for non-employee detection.',
  },
  {
    id: '50',
    type: QuestionType.MCQ,
    question: 'A Marketing Manager at a pet insurance company plans to launch a targeted marketing campaign on social media to acquire new customers. The company has customer profiles, pet profiles, policy information, premiums, and claims data in Amazon Aurora. What steps should be taken to implement a machine learning model to identify potential new customers on social media?',
    options: [
      { id: '50a', text: 'Use regression on customer profile data to understand key characteristics of consumer segments. Find similar profiles on social media.', isCorrect: false },
      { id: '50b', text: 'Use clustering on customer profile data to understand key characteristics of consumer segments. Find similar profiles on social media.', isCorrect: true },
      { id: '50c', text: 'Use a recommendation engine on customer profile data to understand key characteristics of consumer segments. Find similar profiles on social media.', isCorrect: false },
      { id: '50d', text: 'Use a decision tree classifier engine on customer profile data to understand key characteristics of consumer segments. Find similar profiles on social media.', isCorrect: false },
    ],
    explanation: 'Clustering algorithms like K-Means identify natural groupings in customer data based on similar characteristics. Once customer segments are identified, similar profiles can be found on social media for targeted marketing.',
  },
  {
    id: '51',
    type: QuestionType.MCQ,
    question: 'A manufacturing company has a large set of labeled historical sales data. The manufacturer would like to predict how many units of a particular part should be produced each quarter. Which machine learning approach should be used to solve this problem?',
    options: [
      { id: '51a', text: 'Logistic regression', isCorrect: false },
      { id: '51b', text: 'Random Cut Forest (RCF)', isCorrect: false },
      { id: '51c', text: 'Principal component analysis (PCA)', isCorrect: false },
      { id: '51d', text: 'Linear regression', isCorrect: true },
    ],
    explanation: 'The problem requires predicting a continuous numerical value (number of units to produce), which is a regression task. Linear regression is the appropriate algorithm for predicting continuous values based on historical labeled data.',
  },
  {
    id: '52',
    type: QuestionType.MCQ,
    question: 'A financial services company is building a robust serverless data lake on Amazon S3. The data lake should be flexible and meet the following requirements: Support querying old and new data on Amazon S3 through Amazon Athena and Amazon Redshift Spectrum. Support event-driven ETL pipelines. Provide a quick and easy way to understand metadata. Which approach meets these requirements?',
    options: [
      { id: '52a', text: 'Use an AWS Glue crawler to crawl S3 data, an AWS Lambda function to trigger an AWS Glue ETL job, and an AWS Glue Data catalog to search and discover metadata.', isCorrect: true },
      { id: '52b', text: 'Use an AWS Glue crawler to crawl S3 data, an AWS Lambda function to trigger an AWS Batch job, and an external Apache Hive metastore to search and discover metadata.', isCorrect: false },
      { id: '52c', text: 'Use an AWS Glue crawler to crawl S3 data, an Amazon CloudWatch alarm to trigger an AWS Batch job, and an AWS Glue Data Catalog to search and discover metadata.', isCorrect: false },
      { id: '52d', text: 'Use an AWS Glue crawler to crawl S3 data, an Amazon CloudWatch alarm to trigger an AWS Glue ETL job, and an external Apache Hive metastore to search and discover metadata.', isCorrect: false },
    ],
    explanation: 'AWS Glue crawlers automatically discover metadata, the Glue Data Catalog provides centralized metadata management accessible by Athena and Redshift Spectrum, and Lambda functions can trigger event-driven ETL jobs. This combination meets all requirements.',
  },
  {
    id: '53',
    type: QuestionType.MCQ,
    question: "A company's Machine Learning Specialist needs to improve the training speed of a time-series forecasting model using TensorFlow. The training is currently implemented on a single-GPU machine and takes approximately 23 hours to complete. The training needs to be run daily. The model accuracy is acceptable, but the company anticipates a continuous increase in the size of the training data and a need to update the model on an hourly, rather than a daily, basis. The company also wants to minimize coding effort and infrastructure changes. What should the Machine Learning Specialist do to the training solution to allow it to scale for future demand?",
    options: [
      { id: '53a', text: 'Do not change the TensorFlow code. Change the machine to one with a more powerful GPU to speed up the training.', isCorrect: false },
      { id: '53b', text: 'Change the TensorFlow code to implement a Horovod distributed framework supported by Amazon SageMaker. Parallelize the training to as many machines as needed to achieve the business goals.', isCorrect: true },
      { id: '53c', text: 'Switch to using a built-in AWS SageMaker DeepAR model. Parallelize the training to as many machines as needed to achieve the business goals.', isCorrect: false },
      { id: '53d', text: 'Move the training to Amazon EMR and distribute the workload to as many machines as needed to achieve the business goals.', isCorrect: false },
    ],
    explanation: 'Horovod is a distributed training framework that integrates with TensorFlow and is supported by SageMaker. It enables data-parallel training across multiple machines with minimal code changes, allowing horizontal scaling to meet hourly training requirements.',
  },
  {
    id: '54',
    type: QuestionType.MCQ,
    question: 'Which of the following metrics should a Machine Learning Specialist generally use to compare/evaluate machine learning classification models against each other?',
    options: [
      { id: '54a', text: 'Recall', isCorrect: false },
      { id: '54b', text: 'Misclassification rate', isCorrect: false },
      { id: '54c', text: 'Mean absolute percentage error (MAPE)', isCorrect: false },
      { id: '54d', text: 'Area Under the ROC Curve (AUC)', isCorrect: true },
    ],
    explanation: 'AUC (Area Under the ROC Curve) provides a comprehensive measure of model performance across all classification thresholds. It is threshold-independent and works well for comparing different classification models, unlike recall or misclassification rate which are threshold-dependent.',
  },
  {
    id: '55',
    type: QuestionType.MCQ,
    question: 'A company is running a machine learning prediction service that generates 100 TB of predictions every day. A Machine Learning Specialist must generate a visualization of the daily precision-recall curve from the predictions, and forward a read-only version to the Business team. Which solution requires the LEAST coding effort?',
    options: [
      { id: '55a', text: 'Run a daily Amazon EMR workflow to generate precision-recall data, and save the results in Amazon S3. Give the Business team read-only access to S3.', isCorrect: false },
      { id: '55b', text: 'Generate daily precision-recall data in Amazon QuickSight, and publish the results in a dashboard shared with the Business team.', isCorrect: false },
      { id: '55c', text: 'Run a daily Amazon EMR workflow to generate precision-recall data, and save the results in Amazon S3. Visualize the arrays in Amazon QuickSight, and publish them in a dashboard shared with the Business team.', isCorrect: true },
      { id: '55d', text: 'Generate daily precision-recall data in Amazon ES, and publish the results in a dashboard shared with the Business team.', isCorrect: false },
    ],
    explanation: 'EMR provides scalable data processing for generating precision-recall data from 100 TB of predictions. QuickSight can then visualize this data without requiring custom visualization code. This combination minimizes coding effort while providing a shared, read-only dashboard for the Business team.',
  },
  {
    id: '56',
    type: QuestionType.MCQ,
    question: 'A Machine Learning Specialist is preparing data for training on Amazon SageMaker. The Specialist is using one of the SageMaker built-in algorithms for the training. The dataset is stored in .CSV format and is transformed into a numpy.array, which appears to be negatively affecting the speed of the training. What should the Specialist do to optimize the data for training on SageMaker?',
    options: [
      { id: '56a', text: 'Use the SageMaker batch transform feature to transform the training data into a DataFrame.', isCorrect: false },
      { id: '56b', text: 'Use AWS Glue to compress the data into the Apache Parquet format.', isCorrect: false },
      { id: '56c', text: 'Transform the dataset into the RecordIO protobuf format.', isCorrect: true },
      { id: '56d', text: 'Use the SageMaker hyperparameter optimization feature to automatically optimize the data.', isCorrect: false },
    ],
    explanation: 'RecordIO protobuf format is the optimized data format for SageMaker training. It provides better performance than CSV or numpy arrays by enabling efficient streaming and batch access to training data, significantly reducing training time.',
  },
  {
    id: '57',
    type: QuestionType.MCQ,
    question: 'A Machine Learning Specialist is required to build a supervised image-recognition model to identify a cat. The ML Specialist performs some tests and records the following results for a neural network-based image classifier: Total number of images available = 1,000, Test set images = 100 (constant test set). The ML Specialist notices that, in over 75% of the misclassified images, the cats were held upside down by their owners. Which techniques can be used by the ML Specialist to improve this specific test error?',
    options: [
      { id: '57a', text: 'Increase the training data by adding variation in rotation for training images.', isCorrect: true },
      { id: '57b', text: 'Increase the number of epochs for model training', isCorrect: false },
      { id: '57c', text: 'Increase the number of layers for the neural network.', isCorrect: false },
      { id: '57d', text: 'Increase the dropout rate for the second-to-last layer.', isCorrect: false },
    ],
    explanation: 'Data augmentation with rotation variations will help the model learn to recognize cats in different orientations. Since the error is specifically related to upside-down cats, adding rotated training examples is the most effective solution to improve test accuracy for this particular failure mode.',
  },
  {
    id: '58',
    type: QuestionType.MCQ,
    question: 'A Machine Learning Specialist needs to be able to ingest streaming data and store it in Apache Parquet files for exploration and analysis. Which of the following services would both ingest and store this data in the correct format?',
    options: [
      { id: '58a', text: 'AWS DMS', isCorrect: false },
      { id: '58b', text: 'Amazon Kinesis Data Streams', isCorrect: false },
      { id: '58c', text: 'Amazon Kinesis Data Firehose', isCorrect: true },
      { id: '58d', text: 'Amazon Kinesis Data Analytics', isCorrect: false },
    ],
    explanation: 'Amazon Kinesis Data Firehose can ingest streaming data and automatically convert it to Parquet format before delivering to S3. It handles the ingestion and format conversion in a single fully-managed service, making it ideal for this use case.',
  },
  {
    id: '59',
    type: QuestionType.MCQ,
    question: 'A data scientist has explored and sanitized a dataset in preparation for the modeling phase of a supervised learning task. The statistical dispersion can vary widely between features, sometimes by several orders of magnitude. Before moving on to the modeling phase, the data scientist wants to ensure that the prediction performance on the production data is as accurate as possible. Which sequence of steps should the data scientist take to meet these requirements?',
    options: [
      { id: '59a', text: 'Apply random sampling to the dataset. Then split the dataset into training, validation, and test sets.', isCorrect: false },
      { id: '59b', text: 'Split the dataset into training, validation, and test sets. Then rescale the training set and apply the same scaling to the validation and test sets.', isCorrect: true },
      { id: '59c', text: 'Rescale the dataset. Then split the dataset into training, validation, and test sets.', isCorrect: false },
      { id: '59d', text: 'Split the dataset into training, validation, and test sets. Then rescale the training set, the validation set, and the test set independently.', isCorrect: false },
    ],
    explanation: 'Splitting the data first prevents data leakage. The scaling parameters (mean, std) should be computed only from the training set, then applied consistently to validation and test sets. This ensures the model evaluation reflects real-world performance on unseen data.',
  },
  {
    id: '60',
    type: QuestionType.MCQ,
    question: 'A Machine Learning Specialist is assigned a TensorFlow project using Amazon SageMaker for training, and needs to continue working for an extended period with no Wi-Fi access. Which approach should the Specialist use to continue working?',
    options: [
      { id: '60a', text: 'Install Python 3 and boto3 on their laptop and continue the code development using that environment.', isCorrect: false },
      { id: '60b', text: 'Download the TensorFlow Docker container used in Amazon SageMaker from GitHub to their local environment, and use the Amazon SageMaker Python SDK to test the code.', isCorrect: true },
      { id: '60c', text: 'Download TensorFlow from tensorflow.org to emulate the TensorFlow kernel in the SageMaker environment.', isCorrect: false },
      { id: '60d', text: 'Download the SageMaker notebook to their local environment, then install Jupyter Notebooks on their laptop and continue the development in a local notebook.', isCorrect: false },
    ],
    explanation: 'Downloading the SageMaker TensorFlow Docker container allows the Specialist to work locally with the exact same environment as SageMaker. Using the SageMaker Python SDK locally ensures code compatibility when deployed to SageMaker for training.',
  },
  {
    id: '61',
    type: QuestionType.MCQ,
    question: 'A Machine Learning Specialist is working with a large cybersecurity company that manages security events in real time for companies around the world. The cybersecurity company wants to design a solution that will allow it to use machine learning to score malicious events as anomalies on the data as it is being ingested. The company also wants be able to save the results in its data lake for later processing and analysis. What is the MOST efficient way to accomplish these tasks?',
    options: [
      { id: '61a', text: 'Ingest the data using Amazon Kinesis Data Firehose, and use Amazon Kinesis Data Analytics Random Cut Forest (RCF) for anomaly detection. Then use Kinesis Data Firehose to stream the results to Amazon S3.', isCorrect: true },
      { id: '61b', text: 'Ingest the data into Apache Spark Streaming using Amazon EMR, and use Spark MLlib with k-means to perform anomaly detection. Then store the results in an Apache Hadoop Distributed File System (HDFS) using Amazon EMR with a replication factor of three as the data lake.', isCorrect: false },
      { id: '61c', text: 'Ingest the data and store it in Amazon S3. Use AWS Batch along with the AWS Deep Learning AMIs to train a k-means model using TensorFlow on the data in Amazon S3.', isCorrect: false },
      { id: '61d', text: 'Ingest the data and store it in Amazon S3. Have an AWS Glue job that is triggered on demand transform the new data. Then use the built-in Random Cut Forest (RCF) model within Amazon SageMaker to detect anomalies in the data.', isCorrect: false },
    ],
    explanation: 'Kinesis Data Firehose with Kinesis Data Analytics RCF provides real-time anomaly detection for streaming data. This serverless approach efficiently scores events as they are ingested and delivers results directly to S3 for the data lake.',
  },
  {
    id: '62',
    type: QuestionType.MCQ,
    question: 'A Data Scientist wants to gain real-time insights into a data stream of GZIP files. Which solution would allow the use of SQL to query the stream with the LEAST latency?',
    options: [
      { id: '62a', text: 'Amazon Kinesis Data Analytics with an AWS Lambda function to transform the data.', isCorrect: true },
      { id: '62b', text: 'AWS Glue with a custom ETL script to transform the data.', isCorrect: false },
      { id: '62c', text: 'An Amazon Kinesis Client Library to transform the data and save it to an Amazon ES cluster.', isCorrect: false },
      { id: '62d', text: 'Amazon Kinesis Data Firehose to transform the data and put it into an Amazon S3 bucket.', isCorrect: false },
    ],
    explanation: 'Kinesis Data Analytics allows running SQL queries directly on streaming data with Lambda functions for data transformation, providing the lowest latency for real-time insights into GZIP streams.',
  },
  {
    id: '63',
    type: QuestionType.MCQ,
    question: 'A retail company intends to use machine learning to categorize new products. A labeled dataset of current products was provided to the Data Science team. The dataset includes 1,200 products. The labeled dataset has 15 features for each product such as title dimensions, weight, and price. Each product is labeled as belonging to one of six categories such as books, games, electronics, and movies. Which model should be used for categorizing new products using the provided dataset for training?',
    options: [
      { id: '63a', text: 'An XGBoost model where the objective parameter is set to multi:softmax', isCorrect: true },
      { id: '63b', text: 'A deep convolutional neural network (CNN) with a softmax activation function for the last layer', isCorrect: false },
      { id: '63c', text: 'A regression forest where the number of trees is set equal to the number of product categories', isCorrect: false },
      { id: '63d', text: 'A DeepAR forecasting model based on a recurrent neural network (RNN)', isCorrect: false },
    ],
    explanation: 'XGBoost with multi:softmax is ideal for multi-class classification with structured/tabular data. It handles the 15 features efficiently for categorizing products into 6 classes, unlike deep learning approaches which are better suited for image or sequence data.',
  },
  {
    id: '64',
    type: QuestionType.MCQ,
    question: 'A Data Scientist is working on an application that performs sentiment analysis. The validation accuracy is poor, and the Data Scientist thinks that the cause may be a rich vocabulary and a low average frequency of words in the dataset. Which tool should be used to improve the validation accuracy?',
    options: [
      { id: '64a', text: 'Amazon Comprehend syntax analysis and entity detection', isCorrect: false },
      { id: '64b', text: 'Amazon SageMaker BlazingText cbow mode', isCorrect: false },
      { id: '64c', text: 'Natural Language Toolkit (NLTK) stemming and stop word removal', isCorrect: false },
      { id: '64d', text: 'Scikit-learn term frequency-inverse document frequency (TF-IDF) vectorizer', isCorrect: true },
    ],
    explanation: 'TF-IDF vectorization addresses rich vocabulary and low word frequency issues by giving higher weights to distinctive terms and lower weights to common words. This improves feature representation for sentiment analysis with sparse, high-vocabulary text data.',
  },
  {
    id: '65',
    type: QuestionType.MCQ,
    question: 'Machine Learning Specialist is building a model to predict future employment rates based on a wide range of economic factors. While exploring the data, the Specialist notices that the magnitude of the input features vary greatly. The Specialist does not want variables with a larger magnitude to dominate the model. What should the Specialist do to prepare the data for model training?',
    options: [
      { id: '65a', text: 'Apply quantile binning to group the data into categorical bins to keep any relationships in the data by replacing the magnitude with distribution.', isCorrect: false },
      { id: '65b', text: 'Apply the Cartesian product transformation to create new combinations of fields that are independent of the magnitude.', isCorrect: false },
      { id: '65c', text: 'Apply normalization to ensure each field will have a mean of 0 and a variance of 1 to remove any significant magnitude.', isCorrect: true },
      { id: '65d', text: 'Apply the orthogonal sparse bigram (OSB) transformation to apply a fixed-size sliding window to generate new features of a similar magnitude.', isCorrect: false },
    ],
    explanation: 'Normalization (standardization) transforms features to have mean 0 and variance 1, ensuring all features contribute equally to the model regardless of their original magnitude. This prevents features with larger scales from dominating the model predictions.',
  },
  {
    id: '66',
    type: QuestionType.MCQ,
    question: 'A Machine Learning Specialist must build out a process to query a dataset on Amazon S3 using Amazon Athena. The dataset contains more than 800,000 records stored as plaintext CSV files. Each record contains 200 columns and is approximately 1.5 MB in size. Most queries will span 5 to 10 columns only. How should the Machine Learning Specialist transform the dataset to minimize query runtime?',
    options: [
      { id: '66a', text: 'Convert the records to Apache Parquet format.', isCorrect: true },
      { id: '66b', text: 'Convert the records to JSON format.', isCorrect: false },
      { id: '66c', text: 'Convert the records to GZIP CSV format.', isCorrect: false },
      { id: '66d', text: 'Convert the records to XML format.', isCorrect: false },
    ],
    explanation: 'Apache Parquet is a columnar storage format that allows Athena to read only the required columns for each query. This significantly reduces query runtime compared to row-based formats like CSV, JSON, or XML, especially when queries only span 5-10 columns out of 200.',
  },
  {
    id: '67',
    type: QuestionType.MCQ,
    question: 'A Machine Learning Specialist is developing a daily ETL workflow containing multiple ETL jobs. The workflow consists of the following processes: Start the workflow as soon as data is uploaded to Amazon S3. When all the datasets are available in Amazon S3, start an ETL job to join the uploaded datasets with multiple terabyte-sized datasets already stored in Amazon S3. Store the results of joining datasets in Amazon S3. If one of the jobs fails, send a notification to the Administrator. Which configuration will meet these requirements?',
    options: [
      { id: '67a', text: 'Use AWS Lambda to trigger an AWS Step Functions workflow to wait for dataset uploads to complete in Amazon S3. Use AWS Glue to join the datasets. Use an Amazon CloudWatch alarm to send an SNS notification to the Administrator in the case of a failure.', isCorrect: true },
      { id: '67b', text: 'Develop the ETL workflow using AWS Lambda to start an Amazon SageMaker notebook instance. Use a lifecycle configuration script to join the datasets and persist the results in Amazon S3. Use an Amazon CloudWatch alarm to send an SNS notification to the Administrator in the case of a failure.', isCorrect: false },
      { id: '67c', text: 'Develop the ETL workflow using AWS Batch to trigger the start of ETL jobs when data is uploaded to Amazon S3. Use AWS Glue to join the datasets in Amazon S3. Use an Amazon CloudWatch alarm to send an SNS notification to the Administrator in the case of a failure.', isCorrect: false },
      { id: '67d', text: 'Use AWS Lambda to chain other Lambda functions to read and join the datasets in Amazon S3 as soon as the data is uploaded to Amazon S3. Use an Amazon CloudWatch alarm to send an SNS notification to the Administrator in the case of a failure.', isCorrect: false },
    ],
    explanation: 'Step Functions with Lambda provides orchestration for waiting on multiple dataset uploads, Glue handles the terabyte-scale data joining efficiently, and CloudWatch alarms with SNS provide failure notifications. This combination meets all workflow requirements.',
  },
  {
    id: '68',
    type: QuestionType.MSQ,
    question: 'An agency collects census information within a country to determine healthcare and social program needs by province and city. The census form collects responses for approximately 500 questions from each citizen. Which combination of algorithms would provide the appropriate insights? (Choose two.)',
    options: [
      { id: '68a', text: 'The factorization machines (FM) algorithm', isCorrect: false },
      { id: '68b', text: 'The Latent Dirichlet Allocation (LDA) algorithm', isCorrect: false },
      { id: '68c', text: 'The principal component analysis (PCA) algorithm', isCorrect: true },
      { id: '68d', text: 'The k-means algorithm', isCorrect: true },
      { id: '68e', text: 'The Random Cut Forest (RCF) algorithm', isCorrect: false },
    ],
    explanation: 'PCA reduces the 500-question dimensionality to identify key patterns and correlations in census data. K-means clustering segments citizens by geographic regions (province/city) for targeted healthcare and social program insights. Together they provide appropriate census analysis.',
  },
  {
    id: '69',
    type: QuestionType.MCQ,
    question: 'A large consumer goods manufacturer has the following products on sale: 34 different toothpaste variants, 48 different toothbrush variants, 43 different mouthwash variants. The entire sales history of all these products is available in Amazon S3. Currently, the company is using custom-built autoregressive integrated moving average (ARIMA) models to forecast demand for these products. The company wants to predict the demand for a new product that will soon be launched. Which solution should a Machine Learning Specialist apply?',
    options: [
      { id: '69a', text: 'Train a custom ARIMA model to forecast demand for the new product.', isCorrect: false },
      { id: '69b', text: 'Train an Amazon SageMaker DeepAR algorithm to forecast demand for the new product.', isCorrect: true },
      { id: '69c', text: 'Train an Amazon SageMaker k-means clustering algorithm to forecast demand for the new product.', isCorrect: false },
      { id: '69d', text: 'Train a custom XGBoost model to forecast demand for the new product.', isCorrect: false },
    ],
    explanation: 'DeepAR is designed for time series forecasting with multiple related series (product demand across 125 variants). It learns patterns across similar products and can forecast for new products by leveraging the learned dynamics, unlike ARIMA which requires separate models per product.',
  },
  {
    id: '70',
    type: QuestionType.MCQ,
    question: 'A Machine Learning Specialist uploads a dataset to an Amazon S3 bucket protected with server-side encryption using AWS KMS. How should the ML Specialist define the Amazon SageMaker notebook instance so it can read the same dataset from Amazon S3?',
    options: [
      { id: '70a', text: 'Define security group(s) to allow all HTTP inbound/outbound traffic and assign those security group(s) to the Amazon SageMaker notebook instance.', isCorrect: false },
      { id: '70b', text: 'Configure the Amazon SageMaker notebook instance to have access to the VPC. Grant permission in the KMS key policy to the notebook\'s KMS role.', isCorrect: false },
      { id: '70c', text: 'Assign an IAM role to the Amazon SageMaker notebook with S3 read access to the dataset. Grant permission in the KMS key policy to that role.', isCorrect: true },
      { id: '70d', text: 'Assign the same KMS key used to encrypt data in Amazon S3 to the Amazon SageMaker notebook instance.', isCorrect: false },
    ],
    explanation: 'The SageMaker notebook needs an IAM role with S3 read permissions to access the data. Additionally, the KMS key policy must grant the notebook role permission to decrypt the data. This combination enables secure access to KMS-encrypted S3 data from SageMaker.',
  },
  {
    id: '71',
    type: QuestionType.MCQ,
    question: 'A Data Scientist needs to migrate an existing on-premises ETL process to the cloud. The current process runs at regular time intervals and uses PySpark to combine and format multiple large data sources into a single consolidated output for downstream processing. The Data Scientist has been given the following requirements to the cloud solution: Combine multiple data sources. Reuse existing PySpark logic. Run the solution on the existing schedule. Minimize the number of servers that will need to be managed. Which architecture should the Data Scientist use to build this solution?',
    options: [
      { id: '71a', text: 'Write the raw data to Amazon S3. Schedule an AWS Lambda function to submit a Spark step to a persistent Amazon EMR cluster based on the existing schedule. Use the existing PySpark logic to run the ETL job on the EMR cluster. Output the results to a "processed" location in Amazon S3 that is accessible for downstream use.', isCorrect: false },
      { id: '71b', text: 'Write the raw data to Amazon S3. Create an AWS Glue ETL job to perform the ETL processing against the input data. Write the ETL job in PySpark to leverage the existing logic. Create a new AWS Glue trigger to trigger the ETL job based on the existing schedule. Configure the output target of the ETL job to write to a "processed" location in Amazon S3 that is accessible for downstream use.', isCorrect: true },
      { id: '71c', text: 'Write the raw data to Amazon S3. Schedule an AWS Lambda function to run on the existing schedule and process the input data from Amazon S3. Write the Lambda logic in Python and implement the existing PySpark logic to perform the ETL process. Have the Lambda function output the results to a "processed" location in Amazon S3 that is accessible for downstream use.', isCorrect: false },
      { id: '71d', text: 'Use Amazon Kinesis Data Analytics to stream the input data and perform real-time SQL queries against the stream to carry out the required transformations within the stream. Deliver the output results to a "processed" location in Amazon S3 that is accessible for downstream use.', isCorrect: false },
    ],
    explanation: 'AWS Glue provides serverless PySpark execution with managed infrastructure, Glue triggers for scheduling, and direct integration with S3 for both input and output. This minimizes server management while reusing existing PySpark logic.',
  },
  {
    id: '72',
    type: QuestionType.MSQ,
    question: 'A Data Scientist is building a model to predict customer churn using a dataset of 100 continuous numerical features. The Marketing team has not provided any insight about which features are relevant for churn prediction. The Marketing team wants to interpret the model and see the direct impact of relevant features on the model outcome. While training a logistic regression model, the Data Scientist observes that there is a wide gap between the training and validation set accuracy. Which methods can the Data Scientist use to improve the model performance and satisfy the Marketing team\'s needs? (Choose two.)',
    options: [
      { id: '72a', text: 'Add L1 regularization to the classifier', isCorrect: true },
      { id: '72b', text: 'Add features to the dataset', isCorrect: false },
      { id: '72c', text: 'Perform recursive feature elimination', isCorrect: true },
      { id: '72d', text: 'Perform t-distributed stochastic neighbor embedding (t-SNE)', isCorrect: false },
      { id: '72e', text: 'Perform linear discriminant analysis', isCorrect: false },
    ],
    explanation: 'L1 regularization performs feature selection by driving irrelevant feature coefficients to zero, improving model generalization. Recursive feature elimination (RFE) systematically removes least important features, addressing the gap between training and validation accuracy while identifying relevant features for interpretability.',
  },
  {
    id: '73',
    type: QuestionType.MCQ,
    question: 'An aircraft engine manufacturing company is measuring 200 performance metrics in a time-series. Engineers want to detect critical manufacturing defects in near-real time during testing. All of the data needs to be stored for offline analysis. What approach would be the MOST effective to perform near-real time defect detection?',
    options: [
      { id: '73a', text: 'Use AWS IoT Analytics for ingestion, storage, and further analysis. Use Jupyter notebooks from within AWS IoT Analytics to carry out analysis for anomalies.', isCorrect: false },
      { id: '73b', text: 'Use Amazon S3 for ingestion, storage, and further analysis. Use an Amazon EMR cluster to carry out Apache Spark ML k-means clustering to determine anomalies.', isCorrect: false },
      { id: '73c', text: 'Use Amazon S3 for ingestion, storage, and further analysis. Use the Amazon SageMaker Random Cut Forest (RCF) algorithm to determine anomalies.', isCorrect: false },
      { id: '73d', text: 'Use Amazon Kinesis Data Firehose for ingestion and Amazon Kinesis Data Analytics Random Cut Forest (RCF) to perform anomaly detection. Use Kinesis Data Firehose to store data in Amazon S3 for further analysis.', isCorrect: true },
    ],
    explanation: 'Kinesis Data Firehose provides real-time ingestion with automatic Parquet conversion, Kinesis Data Analytics RCF performs streaming anomaly detection, and Firehose delivers results to S3 for offline analysis. This combination enables near-real-time defect detection with persistent storage.',
  },
  {
    id: '74',
    type: QuestionType.MSQ,
    question: 'A Machine Learning team runs its own training algorithm on Amazon SageMaker. The training algorithm requires external assets. The team needs to submit both its own algorithm code and algorithm-specific parameters to Amazon SageMaker. What combination of services should the team use to build a custom algorithm in Amazon SageMaker? (Choose two.)',
    options: [
      { id: '74a', text: 'AWS Secrets Manager', isCorrect: false },
      { id: '74b', text: 'AWS CodeStar', isCorrect: false },
      { id: '74c', text: 'Amazon ECR', isCorrect: true },
      { id: '74d', text: 'Amazon ECS', isCorrect: false },
      { id: '74e', text: 'Amazon S3', isCorrect: true },
    ],
    explanation: 'Amazon ECR hosts the custom algorithm Docker container, while Amazon S3 stores algorithm-specific parameters and external assets. SageMaker pulls the container from ECR and fetches parameters from S3 during training.',
  },
  {
    id: '75',
    type: QuestionType.MCQ,
    question: 'A Machine Learning Specialist wants to determine the appropriate SageMakerVariantInvocationsPerInstance setting for an endpoint automatic scaling configuration. The Specialist has performed a load test on a single instance and determined that peak requests per second (RPS) without service degradation is about 20 RPS. As this is the first deployment, the Specialist intends to set the invocation safety factor to 0.5. Based on the stated parameters and given that the invocations per instance setting is measured on a per-minute basis, what should the Specialist set as the SageMakerVariantInvocationsPerInstance setting?',
    options: [
      { id: '75a', text: '10', isCorrect: false },
      { id: '75b', text: '30', isCorrect: false },
      { id: '75c', text: '600', isCorrect: true },
      { id: '75d', text: '2,400', isCorrect: false },
    ],
    explanation: '20 RPS × 60 seconds = 1,200 requests per minute capacity. With 0.5 safety factor: 1,200 × 0.5 = 600 invocations per instance. This provides a buffer to maintain performance under peak load while preventing over-provisioning.',
  },
  {
    id: '76',
    type: QuestionType.MCQ,
    question: 'A company uses a long short-term memory (LSTM) model to evaluate the risk factors of a particular energy sector. The model reviews multi-page text documents to analyze each sentence of the text and categorize it as either a potential risk or no risk. The model is not performing well, even though the Data Scientist has experimented with many different network structures and tuned the corresponding hyperparameters. Which approach will provide the MAXIMUM performance boost?',
    options: [
      { id: '76a', text: 'Initialize the words by term frequency-inverse document frequency (TF-IDF) vectors pretrained on a large collection of news articles related to the energy sector.', isCorrect: false },
      { id: '76b', text: 'Use gated recurrent units (GRUs) instead of LSTM and run the training process until the validation loss stops decreasing.', isCorrect: false },
      { id: '76c', text: 'Reduce the learning rate and run the training process until the training loss stops decreasing.', isCorrect: false },
      { id: '76d', text: 'Initialize the words by word2vec embeddings pretrained on a large collection of news articles related to the energy sector.', isCorrect: true },
    ],
    explanation: 'Word2Vec embeddings pretrained on energy sector news provide semantic understanding of domain-specific terminology. This initializes the LSTM with meaningful word representations, significantly improving performance compared to random initialization or architectural changes.',
  },
  {
    id: '77',
    type: QuestionType.MSQ,
    question: 'A Machine Learning Specialist needs to move and transform data in preparation for training. Some of the data needs to be processed in near-real time, and other data can be moved hourly. There are existing Amazon EMR MapReduce jobs to clean and feature engineering to perform on the data. Which of the following services can feed data to the MapReduce jobs? (Choose two.)',
    options: [
      { id: '77a', text: 'AWS DMS', isCorrect: false },
      { id: '77b', text: 'Amazon Kinesis', isCorrect: true },
      { id: '77c', text: 'AWS Data Pipeline', isCorrect: true },
      { id: '77d', text: 'Amazon Athena', isCorrect: false },
      { id: '77e', text: 'Amazon ES', isCorrect: false },
    ],
    explanation: 'Amazon Kinesis provides real-time data streaming to EMR for near-real-time processing. AWS Data Pipeline orchestrates data movement and triggers EMR jobs on schedules. Both services can feed data into EMR MapReduce jobs for transformation.',
  },
  {
    id: '78',
    type: QuestionType.MCQ,
    question: 'A Machine Learning Specialist previously trained a logistic regression model using scikit-learn on a local machine, and the Specialist now wants to deploy it to production for inference only. What steps should be taken to ensure Amazon SageMaker can host a model that was trained locally?',
    options: [
      { id: '78a', text: 'Build the Docker image with the inference code. Tag the Docker image with the registry hostname and upload it to Amazon ECR.', isCorrect: true },
      { id: '78b', text: 'Serialize the trained model so the format is compressed for deployment. Tag the Docker image with the registry hostname and upload it to Amazon S3.', isCorrect: false },
      { id: '78c', text: 'Serialize the trained model so the format is compressed for deployment. Build the image and upload it to Docker Hub.', isCorrect: false },
      { id: '78d', text: 'Build the Docker image with the inference code. Configure Docker Hub and upload the image to Amazon ECR.', isCorrect: false },
    ],
    explanation: 'To deploy a local model to SageMaker, you need to: (1) serialize the model in a compatible format, (2) build a Docker image with inference code, (3) tag with ECR registry hostname, and (4) push to Amazon ECR for SageMaker to pull and deploy.',
  },
  {
    id: '79',
    type: QuestionType.MCQ,
    question: 'A trucking company is collecting live image data from its fleet of trucks across the globe. The data is growing rapidly and approximately 100 GB of new data is generated every day. The company wants to explore machine learning uses cases while ensuring the data is only accessible to specific IAM users. Which storage option provides the most processing flexibility and will allow access control with IAM?',
    options: [
      { id: '79a', text: 'Use a database, such as Amazon DynamoDB, to store the images, and set the IAM policies to restrict access to only the desired IAM users.', isCorrect: false },
      { id: '79b', text: 'Use an Amazon S3-backed data lake to store the raw images, and set up the permissions using bucket policies.', isCorrect: true },
      { id: '79c', text: 'Setup up Amazon EMR with Hadoop Distributed File System (HDFS) to store the files, and restrict access to the EMR instances using IAM policies.', isCorrect: false },
      { id: '79d', text: 'Configure Amazon EFS with IAM policies to make the data available to Amazon EC2 instances owned by the IAM users.', isCorrect: false },
    ],
    explanation: 'Amazon S3 provides scalable, cost-effective storage for 100 GB/day of image data. S3 bucket policies combined with IAM policies enable granular access control to specific users while supporting various ML processing frameworks (SageMaker, EMR, Glue) for flexible analysis.',
  },
  {
    id: '80',
    type: QuestionType.MCQ,
    question: 'A credit card company wants to build a credit scoring model to help predict whether a new credit card applicant will default on a credit card payment. The company has collected data from a large number of sources with thousands of raw attributes. Early experiments to train a classification model revealed that many attributes are highly correlated, the large number of features slows down the training speed significantly, and that there are some overfitting issues. The Data Scientist on this project would like to speed up the model training time without losing a lot of information from the original dataset. Which feature engineering technique should the Data Scientist use to meet the objectives?',
    options: [
      { id: '80a', text: 'Run self-correlation on all features and remove highly correlated features', isCorrect: false },
      { id: '80b', text: 'Normalize all numerical values to be between 0 and 1', isCorrect: false },
      { id: '80c', text: 'Use an autoencoder or principal component analysis (PCA) to replace original features with new features', isCorrect: true },
      { id: '80d', text: 'Cluster raw data using k-means and use sample data from each cluster to build a new dataset', isCorrect: false },
    ],
    explanation: 'Autoencoder or PCA dimensionality reduction addresses all three issues: (1) handles correlated features by creating uncorrelated principal components, (2) reduces feature count for faster training, (3) prevents overfitting by eliminating noise. This preserves maximum information while improving training efficiency.',
  },
  {
    id: '81',
    type: QuestionType.MCQ,
    question: 'A Data Scientist is training a multilayer perception (MLP) on a dataset with multiple classes. The target class of interest is unique compared to the other classes within the dataset, but it does not achieve and acceptable recall metric. The Data Scientist has already tried varying the number and size of the MLP\'s hidden layers, which has not significantly improved the results. A solution to improve recall must be implemented as quickly as possible. Which techniques should be used to meet these requirements?',
    options: [
      { id: '81a', text: 'Gather more data using Amazon Mechanical Turk and then retrain', isCorrect: false },
      { id: '81b', text: 'Train an anomaly detection model instead of an MLP', isCorrect: false },
      { id: '81c', text: 'Train an XGBoost model instead of an MLP', isCorrect: false },
      { id: '81d', text: 'Add class weights to the MLP\'s loss function and then retrain', isCorrect: true },
    ],
    explanation: 'Adding class weights to the loss function penalizes misclassification of the rare class more heavily, directly improving recall. This is the fastest solution compared to gathering more data or switching to different algorithms.',
  },
  {
    id: '82',
    type: QuestionType.MCQ,
    question: 'A Machine Learning Specialist works for a credit card processing company and needs to predict which transactions may be fraudulent in near-real time. Specifically, the Specialist must train a model that returns the probability that a given transaction may fraudulent. How should the Specialist frame this business problem?',
    options: [
      { id: '82a', text: 'Streaming classification', isCorrect: false },
      { id: '82b', text: 'Binary classification', isCorrect: true },
      { id: '82c', text: 'Multi-category classification', isCorrect: false },
      { id: '82d', text: 'Regression classification', isCorrect: false },
    ],
    explanation: 'Fraud detection is a binary classification problem (fraudulent vs. legitimate transaction) that requires real-time probability scores. This framing allows the model to output a probability that can be thresholded for fraud alerts.',
  },
  {
    id: '83',
    type: QuestionType.MCQ,
    question: 'A real estate company wants to create a machine learning model for predicting housing prices based on a historical dataset. The dataset contains 32 features. Which model will meet the business requirement?',
    options: [
      { id: '83a', text: 'Logistic regression', isCorrect: false },
      { id: '83b', text: 'Linear regression', isCorrect: true },
      { id: '83c', text: 'K-means', isCorrect: false },
      { id: '83d', text: 'Principal component analysis (PCA)', isCorrect: false },
    ],
    explanation: 'Predicting housing prices is a regression problem (continuous output), making linear regression the appropriate model. The 32 features can be used as independent variables to predict the price target.',
  },
  {
    id: '84',
    type: QuestionType.MCQ,
    question: 'A Machine Learning Specialist is applying a linear least squares regression model to a dataset with 1,000 records and 50 features. Prior to training, the ML Specialist notices that two features are perfectly linearly dependent. Why could this be an issue for the linear least squares regression model?',
    options: [
      { id: '84a', text: 'It could cause the backpropagation algorithm to fail during training', isCorrect: false },
      { id: '84b', text: 'It could create a singular matrix during optimization, which fails to define a unique solution', isCorrect: true },
      { id: '84c', text: 'It could modify the loss function during optimization, causing it to fail during training', isCorrect: false },
      { id: '84d', text: 'It could introduce non-linear dependencies within the data, which could invalidate the linear assumptions of the model', isCorrect: false },
    ],
    explanation: 'Perfect linear dependence creates multicollinearity, resulting in a singular (non-invertible) matrix during the normal equation solution. This prevents finding a unique solution for the regression coefficients.',
  },
  {
    id: '85',
    type: QuestionType.MCQ,
    question: 'Given the following confusion matrix for a movie classification model, what is the true class frequency for Romance and the predicted class frequency for Adventure?',
    options: [
      { id: '85a', text: 'The true class frequency for Romance is 77.56% and the predicted class frequency for Adventure is 20.85%', isCorrect: false },
      { id: '85b', text: 'The true class frequency for Romance is 57.92% and the predicted class frequency for Adventure is 13.12%', isCorrect: true },
      { id: '85c', text: 'The true class frequency for Romance is 0.78 and the predicted class frequency for Adventure is (0.47-0.32)', isCorrect: false },
      { id: '85d', text: 'The true class frequency for Romance is 77.56% × 0.78 and the predicted class frequency for Adventure is 20.85% × 0.32', isCorrect: false },
    ],
    explanation: 'True class frequency for Romance = Romance predictions / Total predictions = 0.5792 (57.92%). Predicted class frequency for Adventure = Adventure predictions / Total predictions = 0.1312 (13.12%). These are calculated from the confusion matrix row and column totals.',
  },
  {
    id: '86',
    type: QuestionType.MCQ,
    question: 'A Machine Learning Specialist wants to bring a custom algorithm to Amazon SageMaker. The Specialist implements the algorithm in a Docker container supported by Amazon SageMaker. How should the Specialist package the Docker container so that Amazon SageMaker can launch the training correctly?',
    options: [
      { id: '86a', text: 'Modify the bash_profile file in the container and add a bash command to start the training program', isCorrect: false },
      { id: '86b', text: 'Use CMD config in the Dockerfile to add the training program as a CMD of the image', isCorrect: false },
      { id: '86c', text: 'Configure the training program as an ENTRYPOINT named train', isCorrect: true },
      { id: '86d', text: 'Copy the training program to directory /opt/ml/train', isCorrect: false },
    ],
    explanation: 'SageMaker requires custom algorithms to have a training entrypoint named "train" configured as the ENTRYPOINT in the Docker container. This allows SageMaker to correctly invoke the training program when launching training jobs.',
  },
  {
    id: '87',
    type: QuestionType.MSQ,
    question: 'A Data Scientist needs to analyze employment data. The dataset contains approximately 10 million observations on people across 10 different features. During the preliminary analysis, the Data Scientist notices that income and age distributions are not normal. While income levels shows a right skew as expected, with fewer individuals having a higher income, the age distribution also shows a right skew, with fewer older individuals participating in the workforce. Which feature transformations can the Data Scientist apply to fix the incorrectly skewed data? (Choose two.)',
    options: [
      { id: '87a', text: 'Cross-validation', isCorrect: false },
      { id: '87b', text: 'Numerical value binning', isCorrect: true },
      { id: '87c', text: 'High-degree polynomial transformation', isCorrect: false },
      { id: '87d', text: 'Logarithmic transformation', isCorrect: true },
      { id: '87e', text: 'One hot encoding', isCorrect: false },
    ],
    explanation: 'Logarithmic transformation effectively reduces right-skewed distributions by compressing large values. Numerical value binning groups continuous values into buckets, reducing the impact of extreme values. Both transformations help normalize skewed data for better model performance.',
  },
  {
    id: '88',
    type: QuestionType.MCQ,
    question: 'A web-based company wants to improve its conversion rate on its landing page. Using a large historical dataset of customer visits, the company has repeatedly trained a multi-class deep learning network algorithm on Amazon SageMaker. However, there is an overfitting problem: training data shows 90% accuracy in predictions, while test data shows 70% accuracy only. The company needs to boost the generalization of its model before deploying it into production to maximize conversions of visits to purchases. Which action is recommended to provide the HIGHEST accuracy model for the company\'s test and validation data?',
    options: [
      { id: '88a', text: 'Increase the randomization of training data in the mini-batches used in training', isCorrect: false },
      { id: '88b', text: 'Allocate a higher proportion of the overall data to the training dataset', isCorrect: false },
      { id: '88c', text: 'Apply L1 or L2 regularization and dropouts to the training', isCorrect: true },
      { id: '88d', text: 'Reduce the number of layers and units (or neurons) from the deep learning network', isCorrect: false },
    ],
    explanation: 'L1/L2 regularization adds penalty terms to the loss function to prevent overfitting, while dropouts randomly deactivate neurons during training to improve generalization. This combination provides the highest accuracy improvement for overfitting deep learning models.',
  },
  {
    id: '89',
    type: QuestionType.MCQ,
    question: 'A Machine Learning Specialist is given a structured dataset on the shopping habits of a company\'s customer base. The dataset contains thousands of columns of data and hundreds of numerical columns for each customer. The Specialist wants to identify whether there are natural groupings for these columns across all customers and visualize the results as quickly as possible. What approach should the Specialist take to accomplish these tasks?',
    options: [
      { id: '89a', text: 'Embed the numerical features using the t-distributed stochastic neighbor embedding (t-SNE) algorithm and create a scatter plot.', isCorrect: true },
      { id: '89b', text: 'Run k-means using the Euclidean distance measure for different values of k and create an elbow plot.', isCorrect: false },
      { id: '89c', text: 'Embed the numerical features using the t-distributed stochastic neighbor embedding (t-SNE) algorithm and create a line graph.', isCorrect: false },
      { id: '89d', text: 'Run k-means using the Euclidean distance measure for different values of k and create box plots for each numerical column within each cluster.', isCorrect: false },
    ],
    explanation: 't-SNE dimensionality reduction preserves local structure and is excellent for visualizing high-dimensional data as 2D/3D scatter plots. This allows quick identification of natural groupings in the customer shopping habits data.',
  },
  {
    id: '90',
    type: QuestionType.MCQ,
    question: 'A Machine Learning Specialist is planning to create a long-running Amazon EMR cluster. The EMR cluster will have 1 master node, 10 core nodes, and 20 task nodes. To save on costs, the Specialist will use Spot Instances in the EMR cluster. Which nodes should the Specialist launch on Spot Instances?',
    options: [
      { id: '90a', text: 'Master node', isCorrect: false },
      { id: '90b', text: 'Any of the core nodes', isCorrect: false },
      { id: '90c', text: 'Any of the task nodes', isCorrect: true },
      { id: '90d', text: 'Both core and task nodes', isCorrect: false },
    ],
    explanation: 'Only task nodes should use Spot Instances. Core nodes store data (HDFS) and their loss would cause job failure. Master node coordinates the cluster and must be reliable. Task nodes are stateless and can be replaced if terminated, making them ideal for Spot Instances.',
  },
  {
    id: '91',
    type: QuestionType.MCQ,
    question: 'A manufacturer of car engines collects data from cars as they are being driven. The data collected includes timestamp, engine temperature, rotations per minute (RPM), and other sensor readings. The company wants to predict when an engine is going to have a problem, so it can notify drivers in advance to get engine maintenance. The engine data is loaded into a data lake for training. Which is the MOST suitable predictive model that can be deployed into production?',
    options: [
      { id: '91a', text: 'Add labels over time to indicate which engine faults occur at what time in the future to turn this into a supervised learning problem. Use a recurrent neural network (RNN) to train the model to recognize when an engine might need maintenance for a certain fault.', isCorrect: true },
      { id: '91b', text: 'This data requires an unsupervised learning algorithm. Use Amazon SageMaker k-means to cluster the data.', isCorrect: false },
      { id: '91c', text: 'Add labels over time to indicate which engine faults occur at what time in the future to turn this into a supervised learning problem. Use a convolutional neural network (CNN) to train the model to recognize when an engine might need maintenance for a certain fault.', isCorrect: false },
      { id: '91d', text: 'This data is already formulated as a time series. Use Amazon SageMaker seq2seq to model the time series.', isCorrect: false },
    ],
    explanation: 'RNNs are ideal for sequential/time-series data like sensor readings because they can learn temporal dependencies. By labeling historical data to indicate when faults occurred, this becomes a supervised learning problem where RNNs can predict future maintenance needs based on patterns in the sensor data.',
  },
  {
    id: '92',
    type: QuestionType.MCQ,
    question: 'A company wants to predict the sale prices of houses based on available historical sales data. The target variable in the company\'s dataset is the sale price. The features include parameters such as the lot size, living area measurements, non-living area measurements, number of bedrooms, number of bathrooms, year built, and postal code. The company wants to use multi-variable linear regression to predict house sale prices. Which step should a machine learning specialist take to remove features that are irrelevant for the analysis and reduce the model\'s complexity?',
    options: [
      { id: '92a', text: 'Plot a histogram of the features and compute their standard deviation. Remove features with high variance.', isCorrect: false },
      { id: '92b', text: 'Plot a histogram of the features and compute their standard deviation. Remove features with low variance.', isCorrect: false },
      { id: '92c', text: 'Build a heatmap showing the correlation of the dataset against itself. Remove features with low mutual correlation scores.', isCorrect: false },
      { id: '92d', text: 'Run a correlation check of all features against the target variable. Remove features with low target variable correlation scores.', isCorrect: true },
    ],
    explanation: 'For regression problems, the most relevant features are those most correlated with the target variable (sale price). Removing features with low correlation to the target reduces complexity while preserving predictive power. This feature selection approach directly measures each feature\'s relationship with what we want to predict.',
  },
  {
    id: '93',
    type: QuestionType.MCQ,
    question: 'A company wants to classify user behavior as either fraudulent or normal. Based on internal research, a machine learning specialist will build a binary classifier based on two features: age of account, denoted by x, and transaction month, denoted by y. The class distributions are illustrated in the provided figure. The positive class is portrayed in red, while the negative class is portrayed in black. Which model would have the HIGHEST accuracy?',
    options: [
      { id: '93a', text: 'Linear support vector machine (SVM)', isCorrect: false },
      { id: '93b', text: 'Decision tree', isCorrect: false },
      { id: '93c', text: 'Support vector machine (SVM) with a radial basis function kernel', isCorrect: true },
      { id: '93d', text: 'Single perceptron with a Tanh activation function', isCorrect: false },
    ],
    explanation: 'Based on the class distributions described (likely non-linear boundaries), an SVM with RBF kernel can create complex, non-linear decision boundaries that separate the fraud and normal classes more accurately than linear models. The RBF kernel maps data to higher dimensions where classes become separable.',
  },
  {
    id: '94',
    type: QuestionType.MSQ,
    question: 'A health care company is planning to use neural networks to classify their X-ray images into normal and abnormal classes. The labeled data is divided into a training set of 1,000 images and a test set of 200 images. The initial training of a neural network model with 50 hidden layers yielded 99% accuracy on the training set, but only 55% accuracy on the test set. What changes should the Specialist consider to solve this issue? (Choose three.)',
    options: [
      { id: '94a', text: 'Choose a higher number of layers', isCorrect: false },
      { id: '94b', text: 'Choose a lower number of layers', isCorrect: true },
      { id: '94c', text: 'Choose a smaller learning rate', isCorrect: false },
      { id: '94d', text: 'Enable dropout', isCorrect: true },
      { id: '94e', text: 'Include all the images from the test set in the training set', isCorrect: false },
      { id: '94f', text: 'Enable early stopping', isCorrect: true },
    ],
    explanation: 'The model is severely overfitting (99% train accuracy vs 55% test accuracy). Reducing layers decreases model capacity and overfitting. Dropout randomly deactivates neurons during training, improving generalization. Early stopping halts training when validation performance degrades, preventing the model from memorizing training data.',
  },
  {
    id: '95',
    type: QuestionType.MCQ,
    question: 'This graph shows the training and validation loss against the epochs for a neural network. The network being trained is as follows: Two dense layers, one output neuron, 100 neurons in each layer, 100 epochs, Random initialization of weights. Which technique can be used to improve model performance in terms of accuracy in the validation set?',
    options: [
      { id: '95a', text: 'Early stopping', isCorrect: true },
      { id: '95b', text: 'Random initialization of weights with appropriate seed', isCorrect: false },
      { id: '95c', text: 'Increasing the number of epochs', isCorrect: false },
      { id: '95d', text: 'Adding another layer with the 100 neurons', isCorrect: false },
    ],
    explanation: 'Early stopping monitors validation loss and stops training when it starts increasing (indicating overfitting begins). This prevents the model from overfitting to training data and improves generalization to the validation set. The graph likely shows validation loss diverging from training loss.',
  },
  {
    id: '96',
    type: QuestionType.MCQ,
    question: 'A Machine Learning Specialist is attempting to build a linear regression model. Given the displayed residual plot only, what is the MOST likely problem with the model?',
    options: [
      { id: '96a', text: 'Linear regression is inappropriate. The residuals do not have constant variance.', isCorrect: true },
      { id: '96b', text: 'Linear regression is inappropriate. The underlying data has outliers.', isCorrect: false },
      { id: '96c', text: 'Linear regression is appropriate. The residuals have a zero mean.', isCorrect: false },
      { id: '96d', text: 'Linear regression is appropriate. The residuals have constant variance.', isCorrect: false },
    ],
    explanation: 'Heteroscedasticity (non-constant variance in residuals) violates linear regression assumptions. The residual plot likely shows a funnel or cone shape where residuals spread out or contract as predicted values increase, indicating the model is inappropriate for this data.',
  },
  {
    id: '97',
    type: QuestionType.MSQ,
    question: 'A large company has developed a BI application that generates reports and dashboards using data collected from various operational metrics. The company wants to provide executives with an enhanced experience so they can use natural language to get data from the reports. The company wants the executives to be able ask questions using written and spoken interfaces. Which combination of services can be used to build this conversational interface? (Choose three.)',
    options: [
      { id: '97a', text: 'Alexa for Business', isCorrect: false },
      { id: '97b', text: 'Amazon Connect', isCorrect: false },
      { id: '97c', text: 'Amazon Lex', isCorrect: true },
      { id: '97d', text: 'Amazon Polly', isCorrect: true },
      { id: '97e', text: 'Amazon Comprehend', isCorrect: false },
      { id: '97f', text: 'Amazon Transcribe', isCorrect: true },
    ],
    explanation: 'Amazon Lex provides the conversational interface for natural language understanding. Amazon Polly converts text responses to speech. Amazon Transcribe converts spoken queries to text. Together they enable both written and spoken natural language interactions with the BI reports.',
  },
  {
    id: '98',
    type: QuestionType.MCQ,
    question: 'A machine learning specialist works for a fruit processing company and needs to build a system that categorizes apples into three types. The specialist has collected a dataset that contains 150 images for each type of apple and applied transfer learning on a neural network that was pretrained on ImageNet with this dataset. The company requires at least 85% accuracy. After an exhaustive grid search, the optimal hyperparameters produced 68% accuracy on the training set and 67% accuracy on the validation set. What can the machine learning specialist do to improve the system\'s accuracy?',
    options: [
      { id: '98a', text: 'Upload the model to an Amazon SageMaker notebook instance and use the Amazon SageMaker HPO feature to optimize the model\'s hyperparameters.', isCorrect: false },
      { id: '98b', text: 'Add more data to the training set and retrain the model using transfer learning to reduce the bias.', isCorrect: true },
      { id: '98c', text: 'Use a neural network model with more layers that are pretrained on ImageNet and apply transfer learning to increase the variance.', isCorrect: false },
      { id: '98d', text: 'Train a new model using the current neural network architecture.', isCorrect: false },
    ],
    explanation: 'The model shows high bias (both train and validation accuracy are low and similar). Adding more training data with transfer learning will help the model learn better features and reduce bias. The low training accuracy indicates underfitting, not overfitting.',
  },
  {
    id: '99',
    type: QuestionType.MCQ,
    question: 'A company uses camera images of the tops of items displayed on store shelves to determine which items were removed and which ones still remain. After several hours of data labeling, the company has a total of 1,000 hand-labeled images covering 10 distinct items. The training results were poor. Which machine learning approach fulfills the company\'s long-term needs?',
    options: [
      { id: '99a', text: 'Convert the images to grayscale and retrain the model', isCorrect: false },
      { id: '99b', text: 'Reduce the number of distinct items from 10 to 2, build the model, and iterate', isCorrect: false },
      { id: '99c', text: 'Attach different colored labels to each item, take the images again, and build the model', isCorrect: false },
      { id: '99d', text: 'Augment training data for each item using image variants like inversions and translations, build the model, and iterate.', isCorrect: true },
    ],
    explanation: 'Data augmentation (inversions, translations, rotations, etc.) artificially expands the training dataset, helping the model generalize better. This is the most scalable approach for long-term needs as it maximizes the value of existing labeled data and improves model robustness without requiring additional manual labeling.',
  },
  {
    id: '100',
    type: QuestionType.MCQ,
    question: 'A Data Scientist is developing a binary classifier to predict whether a patient has a particular disease on a series of test results. The Data Scientist has data on 400 patients randomly selected from the population. The disease is seen in 3% of the population. Which cross-validation strategy should the Data Scientist adopt?',
    options: [
      { id: '100a', text: 'A k-fold cross-validation strategy with k=5', isCorrect: false },
      { id: '100b', text: 'A stratified k-fold cross-validation strategy with k=5', isCorrect: true },
      { id: '100c', text: 'A k-fold cross-validation strategy with k=5 and 3 repeats', isCorrect: false },
      { id: '100d', text: 'An 80/20 stratified split between training and validation', isCorrect: false },
    ],
    explanation: 'Stratified k-fold maintains the class distribution (3% disease, 97% no disease) in each fold. This is critical for imbalanced datasets to ensure each fold representative of the overall population and prevent biased model evaluation.',
  },
  {
    id: '101',
    type: QuestionType.MCQ,
    question: 'A technology startup is using complex deep neural networks and GPU compute to recommend the company\'s products to its existing customers based upon each customer\'s habits and interactions. The solution currently pulls each dataset from an Amazon S3 bucket before loading the data into a TensorFlow model pulled from the company\'s Git repository that runs locally. This job then runs for several hours while continually outputting its progress to the same S3 bucket. The job can be paused, restarted, and continued at any time in the event of a failure, and is run from a central queue. Senior managers are concerned about the complexity of the solution\'s resource management and the costs involved in repeating the process regularly. They ask for the workload to be automated so it runs once a week, starting Monday and completing by the close of business Friday. Which architecture should be used to scale the solution at the lowest cost?',
    options: [
      { id: '101a', text: 'Implement the solution using AWS Deep Learning Containers and run the container as a job using AWS Batch on a GPU-compatible Spot Instance', isCorrect: true },
      { id: '101b', text: 'Implement the solution using a low-cost GPU-compatible Amazon EC2 instance and use the AWS Instance Scheduler to schedule the task', isCorrect: false },
      { id: '101c', text: 'Implement the solution using AWS Deep Learning Containers, run the workload using AWS Fargate running on Spot Instances, and then schedule the task using the built-in task scheduler', isCorrect: false },
      { id: '101d', text: 'Implement the solution using Amazon ECS running on Spot Instances and schedule the task using the ECS service scheduler', isCorrect: false },
    ],
    explanation: 'AWS Batch on GPU Spot Instances provides the lowest cost for batch ML workloads. Deep Learning Containers come pre-configured with frameworks, Spot Instances offer up to 90% savings, and AWS Batch handles job scheduling, scaling, and failure recovery automatically.',
  },
  {
    id: '102',
    type: QuestionType.MCQ,
    question: 'A Machine Learning Specialist prepared the following graph displaying the results of k-means for k = [1..10]: Considering the graph, what is a reasonable selection for the optimal choice of k?',
    options: [
      { id: '102a', text: '1', isCorrect: false },
      { id: '102b', text: '4', isCorrect: true },
      { id: '102c', text: '7', isCorrect: false },
      { id: '102d', text: '10', isCorrect: false },
    ],
    explanation: 'The elbow method selects k at the "elbow" point where diminishing returns begin. Based on the typical elbow curve, k=4 represents the optimal number of clusters where adding more clusters doesn\'t significantly reduce the within-cluster sum of squares (inertia).',
  },
  {
    id: '103',
    type: QuestionType.MCQ,
    question: 'A media company with a very large archive of unlabeled images, text, audio, and video footage wishes to index its assets to allow rapid identification of relevant content by the Research team. The company wants to use machine learning to accelerate the efforts of its in-house researchers who have limited machine learning expertise. Which is the FASTEST route to index the assets?',
    options: [
      { id: '103a', text: 'Use Amazon Rekognition, Amazon Comprehend, and Amazon Transcribe to tag data into distinct categories/classes.', isCorrect: true },
      { id: '103b', text: 'Create a set of Amazon Mechanical Turk Human Intelligence Tasks to label all footage.', isCorrect: false },
      { id: '103c', text: 'Use Amazon Transcribe to convert speech to text. Use the Amazon SageMaker Neural Topic Model (NTM) and Object Detection algorithms to tag data into distinct categories/classes.', isCorrect: false },
      { id: '103d', text: 'Use the AWS Deep Learning AMI and Amazon EC2 GPU instances to create custom models for audio transcription and topic modeling, and use object detection to tag data into distinct categories/classes.', isCorrect: false },
    ],
    explanation: 'Amazon Rekognition, Comprehend, and Transcribe are managed AI services that require no ML expertise and provide immediate results. This is the fastest approach compared to building custom models or using human labeling which takes significantly longer.',
  },
  {
    id: '104',
    type: QuestionType.MCQ,
    question: 'A Machine Learning Specialist is working for an online retailer that wants to run analytics on every customer visit, processed through a machine learning pipeline. The data needs to be ingested by Amazon Kinesis Data Streams at up to 100 transactions per second, and the JSON data blob is 100 KB in size. What is the MINIMUM number of shards in Kinesis Data Streams the Specialist should use to successfully ingest this data?',
    options: [
      { id: '104a', text: '1 shard', isCorrect: false },
      { id: '104b', text: '10 shards', isCorrect: true },
      { id: '104c', text: '100 shards', isCorrect: false },
      { id: '104d', text: '1,000 shards', isCorrect: false },
    ],
    explanation: 'Each Kinesis shard supports: (1) 1MB/second write OR 1000 records/second write. At 100 transactions/second × 100KB = 10MB/second incoming data rate. We need at least 10 shards (10MB/second ÷ 1MB/shard) to handle this throughput.',
  },
  {
    id: '105',
    type: QuestionType.MCQ,
    question: 'A Machine Learning Specialist is deciding between building a naive Bayesian model or a full Bayesian network for a classification problem. The Specialist computes the Pearson correlation coefficients between each feature and finds that their absolute values range between 0.1 to 0.95. Which model describes the underlying data in this situation?',
    options: [
      { id: '105a', text: 'A naive Bayesian model, since the features are all conditionally independent.', isCorrect: false },
      { id: '105b', text: 'A full Bayesian network, since the features are all conditionally independent.', isCorrect: false },
      { id: '105c', text: 'A naive Bayesian model, since some of the features are statistically dependent.', isCorrect: false },
      { id: '105d', text: 'A full Bayesian network, since some of the features are statistically dependent.', isCorrect: true },
    ],
    explanation: 'High correlation (0.1-0.95) between features indicates statistical dependence. Naive Bayes assumes feature independence, which is violated here. A full Bayesian network can model conditional dependencies between features, making it more appropriate for this data.',
  },
  {
    id: '106',
    type: QuestionType.MCQ,
    question: 'A Data Scientist is building a linear regression model and will use resulting p-values to evaluate the statistical significance of each coefficient. Upon inspection of the dataset, the Data Scientist discovers that most of the features are normally distributed. The plot of one feature in the dataset is shown in the graphic. What transformation should the Data Scientist apply to satisfy the statistical assumptions of the linear regression model?',
    options: [
      { id: '106a', text: 'Exponential transformation', isCorrect: false },
      { id: '106b', text: 'Logarithmic transformation', isCorrect: true },
      { id: '106c', text: 'Polynomial transformation', isCorrect: false },
      { id: '106d', text: 'Sinusoidal transformation', isCorrect: false },
    ],
    explanation: 'Logarithmic transformation normalizes right-skewed distributions by compressing large values. This addresses the positive skewness shown in the graphic and helps satisfy linear regression assumptions of normally distributed residuals and constant variance.',
  },
  {
    id: '107',
    type: QuestionType.MCQ,
    question: 'A Machine Learning Specialist is assigned to a Fraud Detection team and must tune an XGBoost model, which is working appropriately for test data. However, with unknown data, it is not working as expected. The existing parameters are provided as follows. Which parameter tuning guidelines should the Specialist follow to avoid overfitting?',
    options: [
      { id: '107a', text: 'Increase the max_depth parameter value.', isCorrect: false },
      { id: '107b', text: 'Lower the max_depth parameter value.', isCorrect: true },
      { id: '107c', text: 'Update the objective to binary:logistic.', isCorrect: false },
      { id: '107d', text: 'Lower the min_child_weight parameter value.', isCorrect: false },
    ],
    explanation: 'Lowering max_depth restricts tree depth, reducing model complexity and overfitting. Smaller trees generalize better to unseen data. XGBoost overfitting is commonly addressed by constraining tree depth, increasing min_child_weight, and adding regularization.',
  },
  {
    id: '108',
    type: QuestionType.MSQ,
    question: 'A data scientist is developing a pipeline to ingest streaming web traffic data. The data scientist needs to implement a process to identify unusual web traffic patterns as part of the pipeline. The patterns will be used downstream for alerting and incident response. The data scientist has access to unlabeled historic data to use, if needed. The solution needs to: Calculate an anomaly score for each web traffic entry. Adapt unusual event identification to changing web patterns over time. Which approach should the data scientist implement to meet these requirements?',
    options: [
      { id: '108a', text: 'Use historic web traffic data to train an anomaly detection model using the Amazon SageMaker Random Cut Forest (RCF) built-in model. Use an Amazon Kinesis Data Stream to process the incoming web traffic data. Attach a preprocessing AWS Lambda function to perform data enrichment by calling the RCF model to calculate the anomaly score for each record.', isCorrect: false },
      { id: '108b', text: 'Use historic web traffic data to train an anomaly detection model using the Amazon SageMaker built-in XGBoost model. Use an Amazon Kinesis Data Stream to process the incoming web traffic data. Attach a preprocessing AWS Lambda function to perform data enrichment by calling the XGBoost model to calculate the anomaly score for each record.', isCorrect: false },
      { id: '108c', text: 'Collect the streaming data using Amazon Kinesis Data Firehose. Map the delivery stream as an input source for Amazon Kinesis Data Analytics. Write a SQL query to run in real time against the streaming data with the k-Nearest Neighbors (kNN) SQL extension to calculate anomaly scores for each record using a tumbling window.', isCorrect: false },
      { id: '108d', text: 'Collect the streaming data using Amazon Kinesis Data Firehose. Map the delivery stream as an input source for Amazon Kinesis Data Analytics. Write a SQL query to run in real time against the streaming data with the Amazon Random Cut Forest (RCF) SQL extension to calculate anomaly scores for each record using a sliding window.', isCorrect: true },
    ],
    explanation: 'Kinesis Data Analytics with RCF SQL extension provides real-time anomaly scoring with automatic adaptation to changing patterns. The sliding window continuously updates the model, and RCF is specifically designed for unsupervised anomaly detection on streaming data.',
  },
  {
    id: '109',
    type: QuestionType.MCQ,
    question: 'A Data Scientist received a set of insurance records, each consisting of a record ID, the final outcome among 200 categories, and the date of the final outcome. Some partial information on claim contents is also provided, but only for a few of the 200 categories. For each outcome category, there are hundreds of records distributed over the past 3 years. The Data Scientist wants to predict how many claims to expect in each category from month to month, a few months in advance. What type of machine learning model should be used?',
    options: [
      { id: '109a', text: 'Classification month-to-month using supervised learning of the 200 categories based on claim contents.', isCorrect: false },
      { id: '109b', text: 'Reinforcement learning using claim IDs and timestamps where the agent will identify how many claims in each category to expect from month to month.', isCorrect: false },
      { id: '109c', text: 'Forecasting using claim IDs and timestamps to identify how many claims in each category to expect from month to month.', isCorrect: true },
      { id: '109d', text: 'Classification with supervised learning of the categories for which partial information on claim contents is provided, and forecasting using claim IDs and timestamps for all other categories.', isCorrect: false },
    ],
    explanation: 'This is a time series forecasting problem - predicting future claim counts based on historical data. The data has temporal patterns (3 years of records) and the goal is to forecast quantities (how many claims) for each category in future months.',
  },
  {
    id: '110',
    type: QuestionType.MCQ,
    question: 'A company that promotes healthy sleep patterns by providing cloud-connected devices currently hosts a sleep tracking application on AWS. The application collects device usage information from device users. The company\'s Data Science team is building a machine learning model to predict if and when a user will stop utilizing the company\'s devices. Predictions from this model are used by a downstream application that determines the best approach for contacting users. The Data Science team is building multiple versions of the machine learning model to evaluate each version against the company\'s business goals. To measure long-term effectiveness, the team wants to run multiple versions of the model in parallel for long periods of time, with the ability to control the portion of inferences served by the models. Which solution satisfies these requirements with MINIMAL effort?',
    options: [
      { id: '110a', text: 'Build and host multiple models in Amazon SageMaker. Create multiple Amazon SageMaker endpoints, one for each model. Programmatically control invoking different models for inference at the application layer.', isCorrect: false },
      { id: '110b', text: 'Build and host multiple models in Amazon SageMaker. Create an Amazon SageMaker endpoint configuration with multiple production variants. Programmatically control the portion of the inferences served by the multiple models by updating the endpoint configuration.', isCorrect: true },
      { id: '110c', text: 'Build and host multiple models in Amazon SageMaker Neo to take into account different types of medical devices. Programmatically control which model is invoked for inference based on the medical device type.', isCorrect: false },
      { id: '110d', text: 'Build and host multiple models in Amazon SageMaker. Create a single endpoint that accesses multiple models. Use Amazon SageMaker batch transform to control invoking the different models through the single endpoint.', isCorrect: false },
    ],
    explanation: 'SageMaker endpoint configurations with multiple production variants allow running models in parallel and controlling traffic distribution with minimal effort. Traffic shifting can be done via API calls to update the endpoint configuration, enabling A/B testing of multiple model versions.',
  },
  {
    id: '111',
    type: QuestionType.MCQ,
    question: 'An agricultural company is interested in using machine learning to detect specific types of weeds in a 100-acre grassland field. Currently, the company uses tractor-mounted cameras to capture multiple images of the field as 10×10 grids. The company also has a large training dataset that consists of annotated images of popular weed classes like broadleaf and non-broadleaf docks. The company wants to build a weed detection model that will detect specific types of weeds and the location of each type within the field. Once the model is ready, it will be hosted on Amazon SageMaker endpoints. The model will perform real-time inferencing using the images captured by the cameras. Which approach should a Machine Learning Specialist take to obtain accurate predictions?',
    options: [
      { id: '111a', text: 'Prepare the images in RecordIO format and upload them to Amazon S3. Use Amazon SageMaker to train, test, and validate the model using an image classification algorithm to categorize images into various weed classes.', isCorrect: false },
      { id: '111b', text: 'Prepare the images in Apache Parquet format and upload them to Amazon S3. Use Amazon SageMaker to train, test, and validate the model using an object-detection single-shot multibox detector (SSD) algorithm.', isCorrect: false },
      { id: '111c', text: 'Prepare the images in RecordIO format and upload them to Amazon S3. Use Amazon SageMaker to train, test, and validate the model using an object-detection single-shot multibox detector (SSD) algorithm.', isCorrect: true },
      { id: '111d', text: 'Prepare the images in Apache Parquet format and upload them to Amazon S3. Use Amazon SageMaker to train, test, and validate the model using an image classification algorithm to categorize images into various weed classes.', isCorrect: false },
    ],
    explanation: 'Object detection with SSD (Single Shot Multibox Detector) is required because the company needs to detect both the type of weed AND its location within the field. Image classification only categorizes entire images. RecordIO is the correct format for SageMaker object detection algorithms.',
  },
  {
    id: '112',
    type: QuestionType.MCQ,
    question: 'A manufacturer is operating a large number of factories with a complex supply chain relationship where unexpected downtime of a machine can cause production to stop at several factories. A data scientist wants to analyze sensor data from the factories to identify equipment in need of preemptive maintenance and then dispatch a service team to prevent unplanned downtime. The sensor readings from a single machine can include up to 200 data points including temperatures, voltages, vibrations, RPMs, and pressure readings. To collect this sensor data, the manufacturer deployed Wi-Fi and LANs across the factories. Even though many factory locations do not have reliable or high-speed internet connectivity, the manufacturer would like to maintain near-real-time inference capabilities. Which deployment architecture for the model will address these business requirements?',
    options: [
      { id: '112a', text: 'Deploy the model in Amazon SageMaker. Run sensor data through this model to predict which machines need maintenance.', isCorrect: false },
      { id: '112b', text: 'Deploy the model on AWS IoT Greengrass in each factory. Run sensor data through this model to infer which machines need maintenance.', isCorrect: true },
      { id: '112c', text: 'Deploy the model to an Amazon SageMaker batch transformation job. Generate inferences in a daily batch report to identify machines that need maintenance.', isCorrect: false },
      { id: '112d', text: 'Deploy the model in Amazon SageMaker and use an IoT rule to write data to an Amazon DynamoDB table. Consume a DynamoDB stream from the table with an AWS Lambda function to invoke the endpoint.', isCorrect: false },
    ],
    explanation: 'AWS IoT Greengrass enables edge inference by deploying ML models directly to local factory devices. This provides near-real-time inference without relying on internet connectivity, addressing the unreliable connectivity requirement while maintaining low-latency predictions.',
  },
  {
    id: '113',
    type: QuestionType.MCQ,
    question: 'A Machine Learning Specialist is designing a scalable data storage solution for Amazon SageMaker. There is an existing TensorFlow-based model implemented as a train.py script that relies on static training data that is currently stored as TFRecords. Which method of providing training data to Amazon SageMaker would meet the business requirements with the LEAST development overhead?',
    options: [
      { id: '113a', text: 'Use Amazon SageMaker script mode and use train.py unchanged. Point the Amazon SageMaker training invocation to the local path of the data without reformatting the training data.', isCorrect: false },
      { id: '113b', text: 'Use Amazon SageMaker script mode and use train.py unchanged. Put the TFRecord data into an Amazon S3 bucket. Point the Amazon SageMaker training invocation to the S3 bucket without reformatting the training data.', isCorrect: true },
      { id: '113c', text: 'Rewrite the train.py script to add a section that converts TFRecords to protobuf and ingests the protobuf data instead of TFRecords.', isCorrect: false },
      { id: '113d', text: 'Prepare the data in the format accepted by Amazon SageMaker. Use AWS Glue or AWS Lambda to reformat and store the data in an Amazon S3 bucket.', isCorrect: false },
    ],
    explanation: 'SageMaker script mode allows using existing train.py scripts without modification. TFRecords are natively supported by TensorFlow, so simply uploading to S3 and pointing SageMaker to the bucket provides the least development overhead while maintaining scalability.',
  },
  {
    id: '114',
    type: QuestionType.MCQ,
    question: 'The chief editor for a product catalog wants the research and development team to build a machine learning system that can be used to detect whether or not individuals in a collection of images are wearing the company\'s retail brand. The team has a set of training data. Which machine learning algorithm should the researchers use that BEST meets their requirements?',
    options: [
      { id: '114a', text: 'Latent Dirichlet Allocation (LDA)', isCorrect: false },
      { id: '114b', text: 'Recurrent neural network (RNN)', isCorrect: false },
      { id: '114c', text: 'K-means', isCorrect: false },
      { id: '114d', text: 'Convolutional neural network (CNN)', isCorrect: true },
    ],
    explanation: 'CNNs are the state-of-the-art for image classification and object detection tasks. They automatically learn spatial hierarchies of features from images, making them ideal for detecting brand logos or clothing items in images.',
  },
  {
    id: '115',
    type: QuestionType.MCQ,
    question: 'A retail company is using Amazon Personalize to provide personalized product recommendations for its customers during a marketing campaign. The company sees a significant increase in sales of recommended items to existing customers immediately after deploying a new solution version, but these sales decrease a short time after deployment. Only historical data from before the marketing campaign is available for training. How should a data scientist adjust the solution?',
    options: [
      { id: '115a', text: 'Use the event tracker in Amazon Personalize to include real-time user interactions.', isCorrect: true },
      { id: '115b', text: 'Add user metadata and use the HRNN-Metadata recipe in Amazon Personalize.', isCorrect: false },
      { id: '115c', text: 'Implement a new solution using the built-in factorization machines (FM) algorithm in Amazon SageMaker.', isCorrect: false },
      { id: '115d', text: 'Add event type and event value fields to the interactions dataset in Amazon Personalize.', isCorrect: false },
    ],
    explanation: 'The event tracker in Amazon Personalize captures real-time user interactions (clicks, purchases, etc.) which are used to continuously improve recommendations. Without real-time feedback, the model relies on stale historical data causing the recommendation quality to degrade over time.',
  },
  {
    id: '116',
    type: QuestionType.MSQ,
    question: 'A machine learning (ML) specialist wants to secure calls to the Amazon SageMaker Service API. The specialist has configured Amazon VPC with a VPC interface endpoint for the Amazon SageMaker Service API and is attempting to secure traffic from specific sets of instances and IAM users. The VPC is configured with a single public subnet. Which combination of steps should the ML specialist take to secure the traffic? (Choose two.)',
    options: [
      { id: '116a', text: 'Add a VPC endpoint policy to allow access to the IAM users.', isCorrect: true },
      { id: '116b', text: 'Modify the users\' IAM policy to allow access to Amazon SageMaker Service API calls only.', isCorrect: false },
      { id: '116c', text: 'Modify the security group on the endpoint network interface to restrict access to the instances.', isCorrect: true },
      { id: '116d', text: 'Modify the ACL on the endpoint network interface to restrict access to the instances.', isCorrect: false },
      { id: '116e', text: 'Add a SageMaker Runtime VPC endpoint interface to the VPC.', isCorrect: false },
    ],
    explanation: 'VPC endpoint policies control which principals can access the endpoint, while security groups on the endpoint network interface can restrict traffic to specific instances. Together they provide defense-in-depth security for SageMaker API access within the VPC.',
  },
  {
    id: '117',
    type: QuestionType.MCQ,
    question: 'An e-commerce company wants to launch a new cloud-based product recommendation feature for its web application. Due to data localization regulations, any sensitive data must not leave its on-premises data center, and the product recommendation model must be trained and tested using nonsensitive data only. Data transfer to the cloud must use IPsec. The web application is hosted on premises with a PostgreSQL database that contains all the data. The company wants the data to be uploaded securely to Amazon S3 each day for model retraining. How should a machine learning specialist meet these requirements?',
    options: [
      { id: '117a', text: 'Create an AWS Glue job to connect to the PostgreSQL DB instance. Ingest tables without sensitive data through an AWS Site-to-Site VPN connection directly into Amazon S3.', isCorrect: true },
      { id: '117b', text: 'Create an AWS Glue job to connect to the PostgreSQL DB instance. Ingest all data through an AWS Site-to-Site VPN connection into Amazon S3 while removing sensitive data using a PySpark job.', isCorrect: false },
      { id: '117c', text: 'Use AWS Database Migration Service (AWS DMS) with table mapping to select PostgreSQL tables with no sensitive data through an SSL connection. Replicate data directly into Amazon S3.', isCorrect: false },
      { id: '117d', text: 'Use PostgreSQL logical replication to replicate all data to PostgreSQL in Amazon EC2 through AWS Direct Connect with a VPN connection. Use AWS Glue to move data from Amazon EC2 to Amazon S3.', isCorrect: false },
    ],
    explanation: 'AWS Site-to-Site VPN provides IPsec encryption for data in transit. AWS Glue can connect to on-premises PostgreSQL, extract only non-sensitive data, and load it to S3. This meets data localization requirements while enabling daily model retraining.',
  },
  {
    id: '118',
    type: QuestionType.MSQ,
    question: 'A logistics company needs a forecast model to predict next month\'s inventory requirements for a single item in 10 warehouses. A machine learning specialist uses Amazon Forecast to develop a forecast model from 3 years of monthly data. There is no missing data. The specialist selects the DeepAR+ algorithm to train a predictor. The predictor\'s mean absolute percentage error (MAPE) is much larger than the MAPE produced by the current human forecasters. Which changes to the CreatePredictor API call could improve the MAPE? (Choose two.)',
    options: [
      { id: '118a', text: 'Set PerformAutoML to true.', isCorrect: true },
      { id: '118b', text: 'Set ForecastHorizon to 4.', isCorrect: false },
      { id: '118c', text: 'Set ForecastFrequency to W for weekly.', isCorrect: false },
      { id: '118d', text: 'Set PerformHPO to true.', isCorrect: true },
      { id: '118e', text: 'Set FeaturizationMethodName to filling.', isCorrect: false },
    ],
    explanation: 'PerformAutoML lets Forecast automatically select the best algorithm and featurization. PerformHPO enables hyperparameter optimization to find optimal settings for DeepAR+. Both improve model accuracy when the default settings underperform.',
  },
  {
    id: '119',
    type: QuestionType.MCQ,
    question: 'A data scientist wants to use Amazon Forecast to build a forecasting model for inventory demand for a retail company. The company has provided a dataset of historic inventory demand for its products as a .csv file stored in an Amazon S3 bucket. The table below shows a sample of the dataset. How should the data scientist transform the data?',
    options: [
      { id: '119a', text: 'Use ETL jobs in AWS Glue to separate the dataset into a target time series dataset and an item metadata dataset. Upload both datasets as .csv files to Amazon S3.', isCorrect: true },
      { id: '119b', text: 'Use a Jupyter notebook in Amazon SageMaker to separate the dataset into a related time series dataset and an item metadata dataset. Upload both datasets as tables in Amazon Aurora.', isCorrect: false },
      { id: '119c', text: 'Use AWS Batch jobs to separate the dataset into a target time series dataset, a related time series dataset, and an item metadata dataset. Upload them directly to Forecast from a local machine.', isCorrect: false },
      { id: '119d', text: 'Use a Jupyter notebook in Amazon SageMaker to transform the data into the optimized protobuf recordIO format. Upload the dataset in this format to Amazon S3.', isCorrect: false },
    ],
    explanation: 'Amazon Forecast requires data in specific formats: target time series (demand history) and optionally item metadata. AWS Glue can transform the raw CSV into these required formats, which are then uploaded to S3 for Forecast consumption.',
  },
  {
    id: '120',
    type: QuestionType.MCQ,
    question: 'A machine learning specialist is running an Amazon SageMaker endpoint using the built-in object detection algorithm on a P3 instance for real-time predictions in a company\'s production application. When evaluating the model\'s resource utilization, the specialist notices that the model is using only a fraction of the GPU. Which architecture changes would ensure that provisioned resources are being utilized effectively?',
    options: [
      { id: '120a', text: 'Redeploy the model as a batch transform job on an M5 instance.', isCorrect: false },
      { id: '120b', text: 'Redeploy the model on an M5 instance. Attach Amazon Elastic Inference to the instance.', isCorrect: true },
      { id: '120c', text: 'Redeploy the model on a P3dn instance.', isCorrect: false },
      { id: '120d', text: 'Deploy the model onto an Amazon Elastic Container Service (Amazon ECS) cluster using a P3 instance.', isCorrect: false },
    ],
    explanation: 'Amazon Elastic Inference allows attaching GPU acceleration to cost-effective CPU instances like M5. This reduces costs by providing the right amount of GPU power needed for the workload rather than paying for an underutilized full P3 instance.',
  },
  {
    id: '121',
    type: QuestionType.MCQ,
    question: 'A data scientist uses an Amazon SageMaker notebook instance to conduct data exploration and analysis. This requires certain Python packages that are not natively available on Amazon SageMaker to be installed on the notebook instance. How can a machine learning specialist ensure that required packages are automatically available on the notebook instance for the data scientist to use?',
    options: [
      { id: '121a', text: 'Install AWS Systems Manager Agent on the underlying Amazon EC2 instance and use Systems Manager Automation to execute the package installation commands.', isCorrect: false },
      { id: '121b', text: 'Create a Jupyter notebook file (.ipynb) with cells containing the package installation commands to execute and place the file under the /etc/init directory of each Amazon SageMaker notebook instance.', isCorrect: false },
      { id: '121c', text: 'Use the conda package manager from within the Jupyter notebook console to apply the necessary conda packages to the default kernel of the notebook.', isCorrect: false },
      { id: '121d', text: 'Create an Amazon SageMaker lifecycle configuration with package installation commands and assign the lifecycle configuration to the notebook instance.', isCorrect: true },
    ],
    explanation: 'SageMaker lifecycle configurations automatically execute scripts when creating or starting notebook instances. This ensures required packages are installed automatically for all users without manual intervention.',
  },
  {
    id: '122',
    type: QuestionType.MCQ,
    question: 'A data scientist needs to identify fraudulent user accounts for a company\'s ecommerce platform. The company wants the ability to determine if a newly created account is associated with a previously known fraudulent user. The data scientist is using AWS Glue to cleanse the company\'s application logs during ingestion. Which strategy will allow the data scientist to identify fraudulent accounts?',
    options: [
      { id: '122a', text: 'Execute the built-in FindDuplicates Amazon Athena query.', isCorrect: false },
      { id: '122b', text: 'Create a FindMatches machine learning transform in AWS Glue.', isCorrect: true },
      { id: '122c', text: 'Create an AWS Glue crawler to infer duplicate accounts in the source data.', isCorrect: false },
      { id: '122d', text: 'Search for duplicate accounts in the AWS Glue Data Catalog.', isCorrect: false },
    ],
    explanation: 'AWS Glue FindMatches ML transform is designed for entity resolution and duplicate detection. It learns to identify matching records even when data quality issues exist, making it ideal for detecting fraudulent accounts linked to known fraudsters.',
  },
  {
    id: '123',
    type: QuestionType.MSQ,
    question: 'A Data Scientist is developing a machine learning model to classify whether a financial transaction is fraudulent. The labeled data available for training consists of 100,000 non-fraudulent observations and 1,000 fraudulent observations. The Data Scientist applies the XGBoost algorithm to the data, resulting in the following confusion matrix when the trained model is applied to a previously unseen validation dataset. The accuracy of the model is 99.1%, but the Data Scientist needs to reduce the number of false negatives. Which combination of steps should the Data Scientist take to reduce the number of false negative predictions by the model? (Choose two.)',
    options: [
      { id: '123a', text: 'Change the XGBoost eval_metric parameter to optimize based on Root Mean Square Error (RMSE).', isCorrect: false },
      { id: '123b', text: 'Increase the XGBoost scale_pos_weight parameter to adjust the balance of positive and negative weights.', isCorrect: true },
      { id: '123c', text: 'Increase the XGBoost max_depth parameter because the model is currently underfitting the data.', isCorrect: false },
      { id: '123d', text: 'Change the XGBoost eval_metric parameter to optimize based on Area Under the ROC Curve (AUC).', isCorrect: true },
      { id: '123e', text: 'Decrease the XGBoost max_depth parameter because the model is currently overfitting the data.', isCorrect: false },
    ],
    explanation: 'scale_pos_weight adjusts class imbalance (1:100 ratio) by penalizing false negatives more heavily. AUC optimizes for ranking and separation between classes, reducing false negatives without adjusting the threshold manually.',
  },
  {
    id: '124',
    type: QuestionType.MCQ,
    question: 'A data scientist has developed a machine learning translation model for English to Japanese by using Amazon SageMaker\'s built-in seq2seq algorithm with 500,000 aligned sentence pairs. While testing with sample sentences, the data scientist finds that the translation quality is reasonable for an example as short as five words. However, the quality becomes unacceptable if the sentence is 100 words long. Which action will resolve the problem?',
    options: [
      { id: '124a', text: 'Change preprocessing to use n-grams.', isCorrect: false },
      { id: '124b', text: 'Add more nodes to the recurrent neural network (RNN) than the largest sentence\'s word count.', isCorrect: false },
      { id: '124c', text: 'Adjust hyperparameters related to the attention mechanism.', isCorrect: true },
      { id: '124d', text: 'Choose a different weight initialization type.', isCorrect: false },
    ],
    explanation: 'The attention mechanism allows the model to focus on relevant parts of the input sequence when generating translations. For long sentences (100 words), the attention mechanism may need tuning (attention horizon, number of layers, or attention type) to maintain translation quality.',
  },
  {
    id: '125',
    type: QuestionType.MSQ,
    question: 'A financial company is trying to detect credit card fraud. The company observed that, on average, 2% of credit card transactions were fraudulent. A data scientist trained a classifier on a year\'s worth of credit card transactions data. The model needs to identify the fraudulent transactions (positives) from the regular ones (negatives). The company\'s goal is to accurately capture as many positives as possible. Which metrics should the data scientist use to optimize the model? (Choose two.)',
    options: [
      { id: '125a', text: 'Specificity', isCorrect: false },
      { id: '125b', text: 'False positive rate', isCorrect: false },
      { id: '125c', text: 'Accuracy', isCorrect: false },
      { id: '125d', text: 'Area under the precision-recall curve', isCorrect: true },
      { id: '125e', text: 'True positive rate', isCorrect: true },
    ],
    explanation: 'For imbalanced fraud detection (2% fraud), True Positive Rate (Recall) measures how many fraudulent transactions are captured. Area under the PR curve is better than ROC for imbalanced data as it focuses on the positive class performance.',
  },
  {
    id: '126',
    type: QuestionType.MCQ,
    question: 'A machine learning specialist is developing a proof of concept for government users whose primary concern is security. The specialist is using Amazon SageMaker to train a convolutional neural network (CNN) model for a photo classifier application. The specialist wants to protect the data so that it cannot be accessed and transferred to a remote host by malicious code accidentally installed on the training container. Which action will provide the MOST secure protection?',
    options: [
      { id: '126a', text: 'Remove Amazon S3 access permissions from the SageMaker execution role.', isCorrect: false },
      { id: '126b', text: 'Encrypt the weights of the CNN model.', isCorrect: false },
      { id: '126c', text: 'Encrypt the training and validation dataset.', isCorrect: false },
      { id: '126d', text: 'Enable network isolation for training jobs.', isCorrect: true },
    ],
    explanation: 'Network isolation (VPC mode) runs training jobs in an isolated VPC with no internet access, preventing data exfiltration by malicious code. This provides the strongest protection against data being transferred to remote hosts.',
  },
  {
    id: '127',
    type: QuestionType.MCQ,
    question: 'A medical imaging company wants to train a computer vision model to detect areas of concern on patients\' CT scans. The company has a large collection of unlabeled CT scans that are linked to each patient and stored in an Amazon S3 bucket. The scans must be accessible to authorized users only. A machine learning engineer needs to build a labeling pipeline. Which set of steps should the engineer take to build the labeling pipeline with the LEAST effort?',
    options: [
      { id: '127a', text: 'Create a workforce with AWS Identity and Access Management (IAM). Build a labeling tool on Amazon EC2 Queue images for labeling by using Amazon Simple Queue Service (Amazon SQS). Write the labeling instructions.', isCorrect: false },
      { id: '127b', text: 'Create an Amazon Mechanical Turk workforce and manifest file. Create a labeling job by using the built-in image classification task type in Amazon SageMaker Ground Truth. Write the labeling instructions.', isCorrect: false },
      { id: '127c', text: 'Create a private workforce and manifest file. Create a labeling job by using the built-in bounding box task type in Amazon SageMaker Ground Truth. Write the labeling instructions.', isCorrect: true },
      { id: '127d', text: 'Create a workforce with Amazon Cognito. Build a labeling web application with AWS Amplify. Build a labeling workflow backend using AWS Lambda. Write the labeling instructions.', isCorrect: false },
    ],
    explanation: 'SageMaker Ground Truth provides built-in bounding box task types for CT scan annotation. Creating a private workforce with managed labeling UI requires the least effort compared to building custom solutions, while maintaining authorized access.',
  },
  {
    id: '128',
    type: QuestionType.MCQ,
    question: 'A company is using Amazon Textract to extract textual data from thousands of scanned text-heavy legal documents daily. The company uses this information to process loan applications automatically. Some of the documents fail business validation and are returned to human reviewers, who investigate the errors. This activity increases the time to process the loan applications. What should the company do to reduce the processing time of loan applications?',
    options: [
      { id: '128a', text: 'Configure Amazon Textract to route low-confidence predictions to Amazon SageMaker Ground Truth. Perform a manual review on those words before performing a business validation.', isCorrect: false },
      { id: '128b', text: 'Use an Amazon Textract synchronous operation instead of an asynchronous operation.', isCorrect: false },
      { id: '128c', text: 'Configure Amazon Textract to route low-confidence predictions to Amazon Augmented AI (Amazon A2I). Perform a manual review on those words before performing a business validation.', isCorrect: true },
      { id: '128d', text: 'Use Amazon Rekognition\'s feature to detect text in an image to extract the data from scanned images. Use this information to process the loan applications.', isCorrect: false },
    ],
    explanation: 'Amazon A2I (Augmented AI) provides built-in human review workflows for Textract. It routes low-confidence predictions directly to human reviewers without building custom infrastructure, reducing processing time while maintaining accuracy.',
  },
  {
    id: '129',
    type: QuestionType.MCQ,
    question: 'A company ingests machine learning (ML) data from web advertising clicks into an Amazon S3 data lake. Click data is added to an Amazon Kinesis data stream by using the Kinesis Producer Library (KPL). The data is loaded into the S3 data lake from the data stream by using an Amazon Kinesis Data Firehose delivery stream. As the data volume increases, an ML specialist notices that the rate of data ingested into Amazon S3 is relatively constant. There also is an increasing backlog of data for Kinesis Data Streams and Kinesis Data Firehose to ingest. Which next step is MOST likely to improve the data ingestion rate into Amazon S3?',
    options: [
      { id: '129a', text: 'Increase the number of S3 prefixes for the delivery stream to write to.', isCorrect: false },
      { id: '129b', text: 'Decrease the retention period for the data stream.', isCorrect: false },
      { id: '129c', text: 'Increase the number of shards for the data stream.', isCorrect: true },
      { id: '129d', text: 'Add more consumers using the Kinesis Client Library (KCL).', isCorrect: false },
    ],
    explanation: 'Kinesis Data Stream throughput is limited by the number of shards (1MB/s write per shard). Increasing shards enables higher ingestion throughput, resolving the backlog issue and improving the data ingestion rate into S3.',
  },
  {
    id: '130',
    type: QuestionType.MCQ,
    question: 'A data scientist must build a custom recommendation model in Amazon SageMaker for an online retail company. Due to the nature of the company\'s products, customers buy only 4-5 products every 5-10 years. So, the company relies on a steady stream of new customers. When a new customer signs up, the company collects data on the customer\'s preferences. How should the data scientist split the dataset into a training and test set for this use case?',
    options: [
      { id: '130a', text: 'Shuffle all interaction data. Split off the last 10% of the interaction data for the test set.', isCorrect: false },
      { id: '130b', text: 'Identify the most recent 10% of interactions for each user. Split off these interactions for the test set.', isCorrect: false },
      { id: '130c', text: 'Identify the 10% of users with the least interaction data. Split off all interaction data from these users for the test set.', isCorrect: false },
      { id: '130d', text: 'Randomly select 10% of the users. Split off all interaction data from these users for the test set.', isCorrect: true },
    ],
    explanation: 'For recommendation systems with sparse purchases, randomly selecting users for the test set ensures the model is evaluated on unseen users (cold start scenario). This simulates the real-world situation of making recommendations to new customers.',
  },
  {
    id: '131',
    type: QuestionType.MSQ,
    question: 'A financial services company wants to adopt Amazon SageMaker as its default data science environment. The company\'s data scientists run machine learning (ML) models on confidential financial data. The company is worried about data egress and wants an ML engineer to secure the environment. Which mechanisms can the ML engineer use to control data egress from SageMaker? (Choose three.)',
    options: [
      { id: '131a', text: 'Connect to SageMaker by using a VPC interface endpoint powered by AWS PrivateLink.', isCorrect: true },
      { id: '131b', text: 'Use SCPs to restrict access to SageMaker.', isCorrect: false },
      { id: '131c', text: 'Disable root access on the SageMaker notebook instances.', isCorrect: false },
      { id: '131d', text: 'Enable network isolation for training jobs and models.', isCorrect: true },
      { id: '131e', text: 'Restrict notebook presigned URLs to specific IPs used by the company.', isCorrect: true },
      { id: '131f', text: 'Protect data with encryption at rest and in transit. Use AWS Key Management Service (AWS KMS) to manage encryption keys.', isCorrect: false },
    ],
    explanation: 'VPC interface endpoints with PrivateLink keep traffic within AWS network. Network isolation prevents internet access. Restricting presigned URLs to company IPs limits access. Together these controls prevent data exfiltration while SCPs, root access disabling, and encryption address different security concerns.',
  },
  {
    id: '132',
    type: QuestionType.MCQ,
    question: 'A company needs to quickly make sense of a large amount of data and gain insight from it. The data is in different formats, the schemas change frequently, and new data sources are added regularly. The company wants to use AWS services to explore multiple data sources, suggest schemas, and enrich and transform the data. The solution should require the least possible coding effort for the data flows and the least possible infrastructure management. Which combination of AWS services will meet these requirements?',
    options: [
      { id: '132a', text: 'Amazon EMR for data discovery, enrichment, and transformation; Amazon Athena for querying and analyzing the results in Amazon S3 using standard SQL; Amazon QuickSight for reporting and getting insights', isCorrect: true },
      { id: '132b', text: 'Amazon Kinesis Data Analytics for data ingestion; Amazon EMR for data discovery, enrichment, and transformation; Amazon Redshift for querying and analyzing the results in Amazon S3', isCorrect: false },
      { id: '132c', text: 'AWS Glue for data discovery, enrichment, and transformation; Amazon Athena for querying and analyzing the results in Amazon S3 using standard SQL; Amazon QuickSight for reporting and getting insights', isCorrect: false },
      { id: '132d', text: 'AWS Data Pipeline for data transfer; AWS Step Functions for orchestrating AWS Lambda jobs for data discovery, enrichment, and transformation; Amazon Athena for querying and analyzing the results in Amazon S3; Amazon QuickSight for reporting and getting insights', isCorrect: false },
    ],
    explanation: 'EMR handles large-scale data processing with minimal management. Athena provides serverless SQL querying on S3 data. QuickSight enables visualization and insights. This combination requires the least coding and infrastructure management while scaling to handle diverse data formats and changing schemas.',
  },
  {
    id: '133',
    type: QuestionType.MCQ,
    question: 'A company is converting a large number of unstructured paper receipts into images. The company wants to create a model based on natural language processing (NLP) to find relevant entities such as date, location, and notes, as well as some custom entities such as receipt numbers. The company is using optical character recognition (OCR) to extract text for data labeling. However, documents are in different structures and formats, and the company is facing challenges with setting up the manual workflows for each document type. Additionally, the company trained a named entity recognition (NER) model for custom entity detection using a small sample size. This model has a very low confidence score and will require retraining with a large dataset. Which solution for text extraction and entity detection will require the LEAST amount of effort?',
    options: [
      { id: '133a', text: 'Extract text from receipt images by using Amazon Textract. Use the Amazon SageMaker BlazingText algorithm to train on the text for entities and custom entities.', isCorrect: false },
      { id: '133b', text: 'Extract text from receipt images by using a deep learning OCR model from the AWS Marketplace. Use the NER deep learning model to extract entities.', isCorrect: false },
      { id: '133c', text: 'Extract text from receipt images by using Amazon Textract. Use Amazon Comprehend for entity detection, and use Amazon Comprehend custom entity recognition for custom entity detection.', isCorrect: true },
      { id: '133d', text: 'Extract text from receipt images by using a deep learning OCR model from the AWS Marketplace. Use Amazon Comprehend for entity detection, and use Amazon Comprehend custom entity recognition for custom entity detection.', isCorrect: false },
    ],
    explanation: 'Amazon Textract provides automatic OCR for various document layouts without custom workflows. Amazon Comprehend offers built-in entity detection and custom entity recognition with minimal ML expertise. This managed solution requires less effort than building custom NER models or using marketplace OCR solutions.',
  },
  {
    id: '134',
    type: QuestionType.MCQ,
    question: 'A company is building a predictive maintenance model based on machine learning (ML). The data is stored in a fully private Amazon S3 bucket that is encrypted at rest with AWS Key Management Service (AWS KMS) CMKs. An ML specialist must run data preprocessing by using an Amazon SageMaker Processing job that is triggered from code in an Amazon SageMaker notebook. The job should read data from Amazon S3, process it, and upload it back to the same S3 bucket. The preprocessing code is stored in a container image in Amazon Elastic Container Registry (Amazon ECR). The ML specialist needs to grant permissions to ensure a smooth data preprocessing workflow. Which set of actions should the ML specialist take to meet these requirements?',
    options: [
      { id: '134a', text: 'Create an IAM role that has permissions to create Amazon SageMaker Processing jobs, S3 read and write access to the relevant S3 bucket, and appropriate KMS and ECR permissions. Attach the role to the SageMaker notebook instance. Create an Amazon SageMaker Processing job from the notebook.', isCorrect: false },
      { id: '134b', text: 'Create an IAM role that has permissions to create Amazon SageMaker Processing jobs. Attach the role to the SageMaker notebook instance. Create an Amazon SageMaker Processing job with an IAM role that has read and write permissions to the relevant S3 bucket, and appropriate KMS and ECR permissions.', isCorrect: true },
      { id: '134c', text: 'Create an IAM role that has permissions to create Amazon SageMaker Processing jobs and to access Amazon ECR. Attach the role to the SageMaker notebook instance. Set up both an S3 endpoint and a KMS endpoint in the default VPC. Create Amazon SageMaker Processing jobs from the notebook.', isCorrect: false },
      { id: '134d', text: 'Create an IAM role that has permissions to create Amazon SageMaker Processing jobs. Attach the role to the SageMaker notebook instance. Set up an S3 endpoint in the default VPC. Create an Amazon SageMaker Processing jobs with the access key and secret key of the IAM user with appropriate KMS and ECR permissions.', isCorrect: false },
    ],
    explanation: 'SageMaker Processing jobs require a separate execution role for the job itself (not the notebook). This role needs S3 access, KMS decryption for encrypted data, and ECR pull permissions. The notebook instance role only needs permission to create processing jobs, following the principle of least privilege.',
  },
  {
    id: '135',
    type: QuestionType.MCQ,
    question: 'A data scientist has been running an Amazon SageMaker notebook instance for a few weeks. During this time, a new version of Jupyter Notebook was released along with additional software updates. The security team mandates that all running SageMaker notebook instances use the latest security and software updates provided by SageMaker. How can the data scientist meet this requirements?',
    options: [
      { id: '135a', text: 'Call the CreateNotebookInstanceLifecycleConfig API operation', isCorrect: false },
      { id: '135b', text: 'Create a new SageMaker notebook instance and mount the Amazon Elastic Block Store (Amazon EBS) volume from the original instance', isCorrect: false },
      { id: '135c', text: 'Stop and then restart the SageMaker notebook instance', isCorrect: true },
      { id: '135d', text: 'Call the UpdateNotebookInstanceLifecycleConfig API operation', isCorrect: false },
    ],
    explanation: 'Stopping and restarting a SageMaker notebook instance triggers SageMaker to update the instance to the latest recommended platform version and install security updates. This is the simplest way to ensure the instance has the latest software without data loss or complex configuration.',
  },
  {
    id: '136',
    type: QuestionType.MCQ,
    question: 'A library is developing an automatic book-borrowing system that uses Amazon Rekognition. Images of library members\' faces are stored in an Amazon S3 bucket. When members borrow books, the Amazon Rekognition CompareFaces API operation compares real faces against the stored faces in Amazon S3. The library needs to improve security by making sure that images are encrypted at rest. Also, when the images are used with Amazon Rekognition, they need to be encrypted in transit. The library also must ensure that the images are not used to improve Amazon Rekognition as a service. How should a machine learning specialist architect the solution to satisfy these requirements?',
    options: [
      { id: '136a', text: 'Enable server-side encryption on the S3 bucket. Submit an AWS Support ticket to opt out of allowing images to be used for improving the service, and follow the process provided by AWS Support.', isCorrect: true },
      { id: '136b', text: 'Switch to using an Amazon Rekognition collection to store the images. Use the IndexFaces and SearchFacesByImage API operations instead of the CompareFaces API operation.', isCorrect: false },
      { id: '136c', text: 'Switch to using the AWS GovCloud (US) Region for Amazon S3 to store images and for Amazon Rekognition to compare faces. Set up a VPN connection and only call the Amazon Rekognition API operations through the VPN.', isCorrect: false },
      { id: '136d', text: 'Enable client-side encryption on the S3 bucket. Set up a VPN connection and only call the Amazon Rekognition API operations through the VPN.', isCorrect: false },
    ],
    explanation: 'S3 server-side encryption protects data at rest. HTTPS encrypts data in transit for Rekognition API calls. Opting out via AWS Support prevents images from being used to improve the Rekognition service. This combination satisfies all security requirements.',
  },
  {
    id: '137',
    type: QuestionType.MCQ,
    question: 'A company is building a line-counting application for use in a quick-service restaurant. The company wants to use video cameras pointed at the line of customers at a given register to measure how many people are in line and deliver notifications to managers if the line grows too long. The restaurant locations have limited bandwidth for connections to external services and cannot accommodate multiple video streams without impacting other operations. Which solution should a machine learning specialist implement to meet these requirements?',
    options: [
      { id: '137a', text: 'Install cameras compatible with Amazon Kinesis Video Streams to stream the data to AWS over the restaurant\'s existing internet connection. Write an AWS Lambda function to take an image and send it to Amazon Rekognition to count the number of faces in the image. Send an Amazon Simple Notification Service (Amazon SNS) notification if the line is too long.', isCorrect: false },
      { id: '137b', text: 'Deploy AWS DeepLens cameras in the restaurant to enable Amazon Rekognition on the AWS DeepLens device, and use it to trigger a local AWS Lambda function when a person is recognized. Use the Lambda function to send an Amazon Simple Notification Service (Amazon SNS) notification if the line is too long.', isCorrect: false },
      { id: '137c', text: 'Build a custom model in Amazon SageMaker to recognize the number of people in an image. Install cameras compatible with Amazon Kinesis Video Streams in the restaurant. Write an AWS Lambda function to take an image. Use the SageMaker endpoint to call the model to count people. Send an Amazon Simple Notification Service (Amazon SNS) notification if the line is too long.', isCorrect: false },
      { id: '137d', text: 'Build a custom model in Amazon SageMaker to recognize the number of people in an image. Deploy AWS DeepLens cameras in the restaurant. Deploy the model to the cameras. Deploy an AWS Lambda function to the cameras to use the model to count people and send an Amazon Simple Notification Service (Amazon SNS) notification if the line is too long.', isCorrect: true },
    ],
    explanation: 'AWS DeepLens allows edge inference by running the model locally on the device. This minimizes bandwidth usage since video isn\'t streamed to the cloud. The custom model counts people and triggers SNS notifications locally, addressing the limited bandwidth constraint.',
  },
  {
    id: '138',
    type: QuestionType.MCQ,
    question: 'A company has set up and deployed its machine learning (ML) model into production with an endpoint using Amazon SageMaker hosting services. The ML team has configured automatic scaling for its SageMaker instances to support workload changes. During testing, the team notices that additional instances are being launched before the new instances are ready. This behavior needs to change as soon as possible. How can the ML team solve this issue?',
    options: [
      { id: '138a', text: 'Decrease the cooldown period for the scale-in activity. Increase the configured maximum capacity of instances.', isCorrect: false },
      { id: '138b', text: 'Replace the current endpoint with a multi-model endpoint using SageMaker.', isCorrect: false },
      { id: '138c', text: 'Set up Amazon API Gateway and AWS Lambda to trigger the SageMaker inference endpoint.', isCorrect: false },
      { id: '138d', text: 'Increase the cooldown period for the scale-out activity.', isCorrect: true },
    ],
    explanation: 'The cooldown period controls how long Auto Scaling waits before starting another scaling activity. Increasing the cooldown period for scale-out prevents new instances from being launched before previous ones are ready and healthy, ensuring instances are fully initialized before receiving traffic.',
  },
  {
    id: '139',
    type: QuestionType.MCQ,
    question: 'A telecommunications company is developing a mobile app for its customers. The company is using an Amazon SageMaker hosted endpoint for machine learning model inferences. Developers want to introduce a new version of the model for a limited number of users who subscribed to a preview feature of the app. After the new version of the model is tested as a preview, developers will evaluate its accuracy. If a new version of the model has better accuracy, developers need to be able to gradually release the new version for all users over a fixed period of time. How can the company implement the testing model with the LEAST amount of operational overhead?',
    options: [
      { id: '139a', text: 'Update the ProductionVariant data type with the new version of the model by using the CreateEndpointConfig operation with the InitialVariantWeight parameter set to 0. Specify the TargetVariant parameter for InvokeEndpoint calls for users who subscribed to the preview feature. When the new version of the model is ready for release, gradually increase InitialVariantWeight until all users have the updated version.', isCorrect: false },
      { id: '139b', text: 'Configure two SageMaker hosted endpoints that serve the different versions of the model. Create an Application Load Balancer (ALB) to route traffic to both endpoints based on the TargetVariant query string parameter. Reconfigure the app to send the TargetVariant query string parameter for users who subscribed to the preview feature. When the new version of the model is ready for release, change the ALB\'s routing algorithm to weighted until all users have the updated version.', isCorrect: false },
      { id: '139c', text: 'Update the DesiredWeightsAndCapacity data type with the new version of the model by using the UpdateEndpointWeightsAndCapacities operation with the DesiredWeight parameter set to 0. Specify the TargetVariant parameter for InvokeEndpoint calls for users who subscribed to the preview feature. When the new version of the model is ready for release, gradually increase DesiredWeight until all users have the updated version.', isCorrect: true },
      { id: '139d', text: 'Configure two SageMaker hosted endpoints that serve the different versions of the model. Create an Amazon Route 53 record that is configured with a simple routing policy and that points to the current version of the model. Configure the mobile app to use the endpoint URL for users who subscribed to the preview feature and to use the Route 53 record for other users. When the new version of the model is ready for release, add a new model version endpoint to Route 53, and switch the policy to weighted until all users have the updated version.', isCorrect: false },
    ],
    explanation: 'UpdateEndpointWeightsAndCapacities allows modifying traffic distribution between model variants without recreating the endpoint. Setting DesiredWeight to 0 initially routes preview users via TargetVariant, then gradually increasing weight implements canary deployments with minimal operational overhead.',
  },
  {
    id: '140',
    type: QuestionType.MCQ,
    question: 'A company offers an online shopping service to its customers. The company wants to enhance the site\'s security by requesting additional information when customers access the site from locations that are different from their normal location. The company wants to update the process to call a machine learning (ML) model to determine when additional information should be requested. The company has several terabytes of data from its existing ecommerce web servers containing the source IP addresses for each request made to the web server. For authenticated requests, the records also contain the login name of the requesting user. Which approach should an ML specialist take to implement the new security feature in the web application?',
    options: [
      { id: '140a', text: 'Use Amazon SageMaker Ground Truth to label each record as either a successful or failed access attempt. Use Amazon SageMaker to train a binary classification model using the factorization machines (FM) algorithm.', isCorrect: false },
      { id: '140b', text: 'Use Amazon SageMaker to train a model using the IP Insights algorithm. Schedule updates and retraining of the model using new log data nightly.', isCorrect: true },
      { id: '140c', text: 'Use Amazon SageMaker Ground Truth to label each record as either a successful or failed access attempt. Use Amazon SageMaker to train a binary classification model using the IP Insights algorithm.', isCorrect: false },
      { id: '140d', text: 'Use Amazon SageMaker to train a model using the Object2Vec algorithm. Schedule updates and retraining of the model using new log data nightly.', isCorrect: false },
    ],
    explanation: 'Amazon SageMaker IP Insights is specifically designed to detect anomalous IP address patterns. It learns typical IP behavior for users and can flag access from unusual locations. Nightly retraining with new log data keeps the model current. This is the most appropriate algorithm for this use case.',
  },
  {
    id: '141',
    type: QuestionType.MCQ,
    question: 'A retail company wants to combine its customer orders with the product description data from its product catalog. The structure and format of the records in each dataset is different. A data analyst tried to use a spreadsheet to combine the datasets, but the effort resulted in duplicate records and records that were not properly combined. The company needs a solution that it can use to combine similar records from the two datasets and remove any duplicates. Which solution will meet these requirements?',
    options: [
      { id: '141a', text: 'Use an AWS Lambda function to process the data. Use two arrays to compare equal strings in the fields from the two datasets and remove any duplicates.', isCorrect: false },
      { id: '141b', text: 'Create AWS Glue crawlers for reading and populating the AWS Glue Data Catalog. Call the AWS Glue SearchTables API operation to perform a fuzzy-matching search on the two datasets, and cleanse the data accordingly.', isCorrect: false },
      { id: '141c', text: 'Create AWS Glue crawlers for reading and populating the AWS Glue Data Catalog. Use the FindMatches transform to cleanse the data.', isCorrect: true },
      { id: '141d', text: 'Create an AWS Lake Formation custom transform. Run a transformation for matching products from the Lake Formation console to cleanse the data automatically.', isCorrect: false },
    ],
    explanation: 'AWS Glue FindMatches ML transform is designed specifically for entity resolution - finding duplicate or matching records across datasets with different formats. It uses machine learning to identify records that should be matched, making it ideal for combining customer orders with product catalog data.',
  },
  {
    id: '142',
    type: QuestionType.MCQ,
    question: 'A company provisions Amazon SageMaker notebook instances for its data science team and creates Amazon VPC interface endpoints to ensure communication between the VPC and the notebook instances. All connections to the Amazon SageMaker API are contained entirely and securely using the AWS network. However, the data science team realizes that individuals outside the VPC can still connect to the notebook instances across the internet. Which set of actions should the data science team take to fix the issue?',
    options: [
      { id: '142a', text: 'Modify the notebook instances\' security group to allow traffic only from the CIDR ranges of the VPC. Apply this security group to all of the notebook instances\' VPC interfaces.', isCorrect: false },
      { id: '142b', text: 'Create an IAM policy that allows the sagemaker:CreatePresignedNotebookInstanceUrl and sagemaker:DescribeNotebookInstance actions from only the VPC endpoints. Apply this policy to all IAM users, groups, and roles used to access the notebook instances.', isCorrect: true },
      { id: '142c', text: 'Add a NAT gateway to the VPC. Convert all of the subnets where the Amazon SageMaker notebook instances are hosted to private subnets. Stop and start all of the notebook instances to reassign only private IP addresses.', isCorrect: false },
      { id: '142d', text: 'Change the network ACL of the subnet the notebook is hosted in to restrict access to anyone outside the VPC.', isCorrect: false },
    ],
    explanation: 'IAM policies can restrict which VPC endpoints can invoke SageMaker API actions. By requiring sagemaker:CreatePresignedNotebookInstanceUrl and sagemaker:DescribeNotebookInstance to come only from VPC endpoints, you prevent presigned URL generation from outside the VPC, effectively blocking external access.',
  },
  {
    id: '143',
    type: QuestionType.MCQ,
    question: 'A company will use Amazon SageMaker to train and host a machine learning (ML) model for a marketing campaign. The majority of data is sensitive customer data. The data must be encrypted at rest. The company wants AWS to maintain the root of trust for the master keys and wants encryption key usage to be logged. Which implementation will meet these requirements?',
    options: [
      { id: '143a', text: 'Use encryption keys that are stored in AWS Cloud HSM to encrypt the ML data volumes, and to encrypt the model artifacts and data in Amazon S3.', isCorrect: false },
      { id: '143b', text: 'Use SageMaker built-in transient keys to encrypt the ML data volumes. Enable default encryption for new Amazon Elastic Block Store (Amazon EBS) volumes.', isCorrect: false },
      { id: '143c', text: 'Use customer managed keys in AWS Key Management Service (AWS KMS) to encrypt the ML data volumes, and to encrypt the model artifacts and data in Amazon S3.', isCorrect: true },
      { id: '143d', text: 'Use AWS Security Token Service (AWS STS) to create temporary tokens to encrypt the ML storage volumes, and to encrypt the model artifacts and data in Amazon S3.', isCorrect: false },
    ],
    explanation: 'Customer managed keys in AWS KMS provide: AWS-managed root of trust for master keys, full control over key policies, detailed logging of key usage via CloudTrail, and compliance with encryption requirements for both EBS volumes and S3 data.',
  },
  {
    id: '144',
    type: QuestionType.MCQ,
    question: 'A machine learning specialist stores IoT soil sensor data in Amazon DynamoDB table and stores weather event data as JSON files in Amazon S3. The dataset in DynamoDB is 10 GB in size and the dataset in Amazon S3 is 5 GB in size. The specialist wants to train a model on this data to help predict soil moisture levels as a function of weather events using Amazon SageMaker. Which solution will accomplish the necessary transformation to train the Amazon SageMaker model with the LEAST amount of administrative overhead?',
    options: [
      { id: '144a', text: 'Launch an Amazon EMR cluster. Create an Apache Hive external table for the DynamoDB table and S3 data. Join the Hive tables and write the results out to Amazon S3.', isCorrect: false },
      { id: '144b', text: 'Crawl the data using AWS Glue crawlers. Write an AWS Glue ETL job that merges the two tables and writes the output to an Amazon Redshift cluster.', isCorrect: false },
      { id: '144c', text: 'Enable Amazon DynamoDB Streams on the sensor table. Write an AWS Lambda function that consumes the stream and appends the results to the existing weather files in Amazon S3.', isCorrect: false },
      { id: '144d', text: 'Crawl the data using AWS Glue crawlers. Write an AWS Glue ETL job that merges the two tables and writes the output in CSV format to Amazon S3.', isCorrect: true },
    ],
    explanation: 'AWS Glue crawlers automatically discover data schemas. AWS Glue ETL jobs can read from both DynamoDB and S3, join the datasets, and write to S3 in CSV format for SageMaker training. This serverless approach requires minimal infrastructure management and administrative overhead.',
  },
  {
    id: '145',
    type: QuestionType.MCQ,
    question: 'A company sells thousands of products on a public website and wants to automatically identify products with potential durability problems. The company has 1,000 reviews with date, star rating, review text, review summary, and customer email fields, but many reviews are incomplete and have empty fields. Each review has already been labeled with the correct durability result. A machine learning specialist must train a model to identify reviews expressing concerns over product durability. The first model needs to be trained and ready to review in 2 days. What is the MOST direct approach to solve this problem within 2 days?',
    options: [
      { id: '145a', text: 'Train a custom classifier by using Amazon Comprehend.', isCorrect: true },
      { id: '145b', text: 'Build a recurrent neural network (RNN) in Amazon SageMaker by using Gluon and Apache MXNet.', isCorrect: false },
      { id: '145c', text: 'Train a built-in BlazingText model using Word2Vec mode in Amazon SageMaker.', isCorrect: false },
      { id: '145d', text: 'Use a built-in seq2seq model in Amazon SageMaker.', isCorrect: false },
    ],
    explanation: 'Amazon Comprehend custom classifier is a fully managed NLP service that can train a text classification model with minimal effort. Given the 2-day deadline and labeled dataset, this is the most direct approach - no model architecture design, training infrastructure, or ML expertise required.',
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
