import { useState, useCallback, useMemo } from 'react';
import type {
  Question,
  UserAnswer,
  QuizResults,
} from '../types/quiz';
import {
  QuizState,
  QuestionType,
} from '../types/quiz';

export interface UseQuizOptions {
  questions: Question[];
  onQuestionChange?: (index: number) => void;
  onComplete?: (results: QuizResults) => void;
}

/**
 * useQuiz Hook
 * 
 * Custom hook for managing quiz state and flow.
 * Handles question navigation, answer submission, and result calculation.
 */
export const useQuiz = ({
  questions,
  onQuestionChange,
  onComplete,
}: UseQuizOptions) => {
  const [quizState, setQuizState] = useState<QuizState>(QuizState.START);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [answers, setAnswers] = useState<Map<string, UserAnswer>>(new Map());
  const [startTime, setStartTime] = useState<number | undefined>();

  const currentQuestion = questions[currentQuestionIndex];
  const totalQuestions = questions.length;

  /**
   * Start the quiz
   */
  const startQuiz = useCallback(() => {
    setQuizState(QuizState.QUESTION);
    setStartTime(Date.now());
    setCurrentQuestionIndex(0);
    setAnswers(new Map());
  }, []);

  /**
   * Navigate to a specific question
   */
  const goToQuestion = useCallback(
    (index: number) => {
      if (index < 0 || index >= totalQuestions) return;
      setCurrentQuestionIndex(index);
      onQuestionChange?.(index);
    },
    [totalQuestions, onQuestionChange]
  );

  /**
   * Navigate to the next question
   */
  const goToNextQuestion = useCallback(() => {
    goToQuestion(currentQuestionIndex + 1);
  }, [currentQuestionIndex, goToQuestion]);

  /**
   * Navigate to the previous question
   */
  const goToPreviousQuestion = useCallback(() => {
    goToQuestion(currentQuestionIndex - 1);
  }, [currentQuestionIndex, goToQuestion]);

  /**
   * Select or deselect an option for the current question
   */
  const toggleOption = useCallback(
    (optionId: string) => {
      if (!currentQuestion) return;

      setAnswers((prevAnswers) => {
        const newAnswers = new Map(prevAnswers);
        const existingAnswer = newAnswers.get(currentQuestion.id);

        let selectedOptionIds: string[];

        if (currentQuestion.type === QuestionType.MCQ) {
          // For MCQ, selecting a new option replaces the previous selection
          selectedOptionIds = [optionId];
        } else {
          // For MSQ, toggle the option
          if (existingAnswer) {
            selectedOptionIds = existingAnswer.selectedOptionIds.includes(optionId)
              ? existingAnswer.selectedOptionIds.filter((id) => id !== optionId)
              : [...existingAnswer.selectedOptionIds, optionId];
          } else {
            selectedOptionIds = [optionId];
          }
        }

        // Check if the answer is correct
        const correctOptionIds = currentQuestion.options
          .filter((opt) => opt.isCorrect)
          .map((opt) => opt.id);

        const isCorrect =
          selectedOptionIds.length === correctOptionIds.length &&
          selectedOptionIds.every((id) => correctOptionIds.includes(id));

        newAnswers.set(currentQuestion.id, {
          questionId: currentQuestion.id,
          selectedOptionIds,
          isCorrect,
        });

        return newAnswers;
      });
    },
    [currentQuestion]
  );

  /**
   * Submit the current answer
   */
  const submitAnswer = useCallback(() => {
    if (!currentQuestion) return;

    setAnswers((prevAnswers) => {
      const newAnswers = new Map(prevAnswers);
      const answer = newAnswers.get(currentQuestion.id);

      if (!answer) {
        // If no answer was selected, create an empty answer
        newAnswers.set(currentQuestion.id, {
          questionId: currentQuestion.id,
          selectedOptionIds: [],
          isCorrect: false,
        });
      }

      return newAnswers;
    });

    setQuizState(QuizState.FEEDBACK);
  }, [currentQuestion]);

  /**
   * Finish the quiz and calculate results
   */
  const finishQuiz = useCallback(() => {
    const quizEndTime = Date.now();
    setQuizState(QuizState.RESULTS);

    setAnswers((currentAnswers) => {
      const results: QuizResults = {
        totalQuestions,
        correctAnswers: 0,
        incorrectAnswers: 0,
        skippedQuestions: 0,
        score: 0,
        timeTaken: startTime ? Math.round((quizEndTime - startTime) / 1000) : 0,
        answers: Array.from(currentAnswers.values()),
      };

      results.answers.forEach((answer) => {
        if (answer.selectedOptionIds.length === 0) {
          results.skippedQuestions++;
        } else if (answer.isCorrect) {
          results.correctAnswers++;
        } else {
          results.incorrectAnswers++;
        }
      });

      results.score = Math.round(
        (results.correctAnswers / results.totalQuestions) * 100
      );

      setTimeout(() => {
        onComplete?.(results);
      }, 0);

      return currentAnswers;
    });
  }, [startTime, totalQuestions, onComplete]);

  /**
   * Move to the next question after viewing feedback
   */
  const nextQuestion = useCallback(() => {
    if (currentQuestionIndex < totalQuestions - 1) {
      setQuizState(QuizState.QUESTION);
      goToNextQuestion();
    } else {
      finishQuiz();
    }
  }, [currentQuestionIndex, totalQuestions, goToNextQuestion, finishQuiz]);

  /**
   * Restart the quiz
   */
  const restartQuiz = useCallback(() => {
    setQuizState(QuizState.START);
    setCurrentQuestionIndex(0);
    setAnswers(new Map());
    setStartTime(undefined);
  }, []);

  /**
   * Get the current answer for the current question
   */
  const currentAnswer = useMemo(() => {
    if (!currentQuestion) return undefined;
    return answers.get(currentQuestion.id);
  }, [answers, currentQuestion]);

  /**
   * Check if the current question has been answered
   */
  const isQuestionAnswered = useMemo(() => {
    if (!currentQuestion) return false;
    const answer = answers.get(currentQuestion.id);
    return answer !== undefined && answer.selectedOptionIds.length > 0;
  }, [answers, currentQuestion]);

  /**
   * Get the number of answered questions
   */
  const answeredCount = useMemo(() => {
    return Array.from(answers.values()).filter(
      (answer) => answer.selectedOptionIds.length > 0
    ).length;
  }, [answers]);

  /**
   * Check if all questions have been answered
   */
  const isAllAnswered = useMemo(() => {
    return answeredCount === totalQuestions;
  }, [answeredCount, totalQuestions]);

  return {
    // State
    quizState,
    currentQuestion,
    currentQuestionIndex,
    totalQuestions,
    answers,
    currentAnswer,
    isQuestionAnswered,
    answeredCount,
    isAllAnswered,
    
    // Actions
    startQuiz,
    goToQuestion,
    goToNextQuestion,
    goToPreviousQuestion,
    toggleOption,
    submitAnswer,
    nextQuestion,
    finishQuiz,
    restartQuiz,
  };
};

export default useQuiz;
