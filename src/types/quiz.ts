/**
 * Question Types
 */
export const QuestionType = {
  MCQ: 'mcq',
  MSQ: 'msq',
} as const;

export type QuestionType = typeof QuestionType[keyof typeof QuestionType];

/**
 * Option state for UI rendering
 */
export const OptionState = {
  DEFAULT: 'default',
  HOVER: 'hover',
  SELECTED: 'selected',
  CORRECT: 'correct',
  INCORRECT: 'incorrect',
} as const;

export type OptionState = typeof OptionState[keyof typeof OptionState];

/**
 * Quiz flow states
 */
export const QuizState = {
  START: 'start',
  QUESTION: 'question',
  SUBMITTED: 'submitted',
  FEEDBACK: 'feedback',
  RESULTS: 'results',
} as const;

export type QuizState = typeof QuizState[keyof typeof QuizState];

/**
 * Individual option for a question
 */
export interface Option {
  id: string;
  text: string;
  isCorrect: boolean;
}

/**
 * Question data structure
 */
export interface Question {
  id: string;
  type: QuestionType;
  question: string;
  options: Option[];
  explanation?: string;
  timeLimit?: number; // Optional time limit in seconds
}

/**
 * User's answer for a question
 */
export interface UserAnswer {
  questionId: string;
  selectedOptionIds: string[];
  isCorrect: boolean;
  timeTaken?: number; // Time taken to answer in seconds
}

/**
 * Quiz session data
 */
export interface QuizSession {
  questions: Question[];
  currentQuestionIndex: number;
  answers: Map<string, UserAnswer>;
  state: QuizState;
  startTime?: number;
  endTime?: number;
}

/**
 * Quiz results summary
 */
export interface QuizResults {
  totalQuestions: number;
  correctAnswers: number;
  incorrectAnswers: number;
  skippedQuestions: number;
  score: number; // Percentage (0-100)
  timeTaken: number; // Total time in seconds
  answers: UserAnswer[];
}
