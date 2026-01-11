import React, { useState, useEffect } from 'react';
import type { Question, UserAnswer } from '../../types/quiz';
import { QuestionType, OptionState } from '../../types/quiz';

export interface QuestionCardProps {
  question: Question;
  selectedOptionIds: string[];
  currentQuestionIndex?: number;
  totalQuestions?: number;
  showFeedback?: boolean;
  userAnswer?: UserAnswer;
  onOptionToggle: (optionId: string) => void;
  onSubmit?: () => void;
  onNext?: () => void;
  isSubmitting?: boolean;
}

const optionStateStyles: Record<OptionState, { bg: string; border: string; text: string }> = {
  [OptionState.DEFAULT]: { bg: 'white', border: '#e2e8f0', text: '#334155' },
  [OptionState.HOVER]: { bg: '#f8fafc', border: '#cbd5e1', text: '#334155' },
  [OptionState.SELECTED]: { bg: '#0f172a', border: '#0f172a', text: 'white' },
  [OptionState.CORRECT]: { bg: '#ecfdf5', border: '#10b981', text: '#065f46' },
  [OptionState.INCORRECT]: { bg: '#fef2f2', border: '#ef4444', text: '#991b1b' },
};

const indicatorStateStyles: Record<OptionState, { border: string; bg: string }> = {
  [OptionState.DEFAULT]: { border: '#cbd5e1', bg: 'white' },
  [OptionState.HOVER]: { border: '#94a3b8', bg: 'white' },
  [OptionState.SELECTED]: { border: '#0f172a', bg: '#0f172a' },
  [OptionState.CORRECT]: { border: '#10b981', bg: '#10b981' },
  [OptionState.INCORRECT]: { border: '#ef4444', bg: '#ef4444' },
};

export const QuestionCard: React.FC<QuestionCardProps> = ({
  question,
  selectedOptionIds,
  currentQuestionIndex = 1,
  totalQuestions = 1,
  showFeedback = false,
  userAnswer,
  onOptionToggle,
  onSubmit,
  onNext,
  isSubmitting = false,
}) => {
  const [hoveredOptionId, setHoveredOptionId] = useState<string | null>(null);
  const [isMobile, setIsMobile] = useState(false);
  const [isTablet, setIsTablet] = useState(false);

  useEffect(() => {
    const checkMedia = () => {
      setIsMobile(window.matchMedia('(max-width: 768px)').matches);
      setIsTablet(window.matchMedia('(max-width: 1024px)').matches);
    };
    checkMedia();
    const mobileQuery = window.matchMedia('(max-width: 768px)');
    const tabletQuery = window.matchMedia('(max-width: 1024px)');
    mobileQuery.addEventListener('change', checkMedia);
    tabletQuery.addEventListener('change', checkMedia);
    return () => {
      mobileQuery.removeEventListener('change', checkMedia);
      tabletQuery.removeEventListener('change', checkMedia);
    };
  }, []);

  const padding = isMobile ? '16px' : isTablet ? '24px' : '48px';
  const gap = isMobile ? '24px' : isTablet ? '40px' : '64px';
  const headerPadding = isMobile ? '12px 16px' : '16px 24px';
  const actionPadding = isMobile ? '16px' : '20px 24px';
  const titleSize = isMobile ? '18px' : '20px';
  const optionPadding = isMobile ? '14px' : '16px 20px';

  const getOptionState = (optionId: string): OptionState => {
    const isHovered = hoveredOptionId === optionId;
    const isSelected = selectedOptionIds.includes(optionId);

    if (showFeedback && userAnswer) {
      const option = question.options.find((opt) => opt.id === optionId);
      if (!option) return OptionState.DEFAULT;
      if (option.isCorrect) return OptionState.CORRECT;
      if (isSelected && !option.isCorrect) return OptionState.INCORRECT;
      return OptionState.DEFAULT;
    }

    if (isSelected) return OptionState.SELECTED;
    if (isHovered) return OptionState.HOVER;
    return OptionState.DEFAULT;
  };

  const isMCQ = question.type === QuestionType.MCQ;
  const isMSQ = question.type === QuestionType.MSQ;
  const progressPercent = Math.round(((currentQuestionIndex - 1) / totalQuestions) * 100);

  return (
    <div style={{ minHeight: 'calc(100vh - 65px)', display: 'flex', flexDirection: 'column' }}>
      <div style={{
        backgroundColor: 'white',
        borderBottom: '1px solid #e2e8f0',
        padding: headerPadding,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: isMobile ? '8px' : '16px', maxWidth: '1200px', margin: '0 auto' }}>
          <span style={{ fontSize: isMobile ? '12px' : '13px', fontWeight: '500', color: '#64748b', minWidth: isMobile ? '80px' : '100px' }}>
            Question {currentQuestionIndex} of {totalQuestions}
          </span>
          <div style={{ flex: 1, height: isMobile ? '4px' : '6px', backgroundColor: '#e2e8f0', borderRadius: '3px', overflow: 'hidden' }}>
            <div style={{
              width: `${progressPercent}%`,
              height: '100%',
              backgroundColor: '#0f172a',
              borderRadius: '3px',
              transition: 'width 0.3s ease',
            }} />
          </div>
          <span style={{ fontSize: isMobile ? '12px' : '13px', fontWeight: '600', color: '#0f172a', minWidth: isMobile ? '40px' : '45px', textAlign: 'right' }}>
            {progressPercent}%
          </span>
        </div>
      </div>

      <div style={{
        flex: 1,
        display: 'grid',
        gridTemplateColumns: isMobile ? '1fr' : '1fr 1fr',
        gap: gap,
        padding: padding,
        maxWidth: '1200px',
        margin: '0 auto',
        width: '100%',
      }}>
        <div>
          <div style={{ marginBottom: isMobile ? '16px' : '20px' }}>
            <span style={{
              display: 'inline-block',
              padding: isMobile ? '4px 10px' : '6px 12px',
              backgroundColor: '#f1f5f9',
              color: '#475569',
              fontSize: isMobile ? '11px' : '12px',
              fontWeight: '500',
              borderRadius: '4px',
              marginBottom: isMobile ? '12px' : '16px',
            }}>
              {isMCQ ? 'Single Answer' : 'Multiple Select'}
            </span>
            <h2 style={{
              fontSize: titleSize,
              fontWeight: '500',
              color: '#0f172a',
              lineHeight: '1.55',
            }}>
              {question.question}
            </h2>
          </div>

          {isMSQ && !showFeedback && (
            <div style={{
              padding: isMobile ? '10px 14px' : '12px 16px',
              backgroundColor: '#f8fafc',
              borderRadius: '6px',
              border: '1px solid #e2e8f0',
            }}>
              <p style={{ fontSize: isMobile ? '12px' : '13px', color: '#64748b', margin: 0 }}>
                Select all that apply
              </p>
            </div>
          )}
        </div>

        <div style={{ display: 'flex', flexDirection: 'column' }}>
          {question.options.map((option, index) => {
            const state = getOptionState(option.id);
            const styles = optionStateStyles[state];
            const indicatorStyles = indicatorStateStyles[state];

            return (
              <button
                key={option.id}
                onClick={() => !showFeedback && onOptionToggle(option.id)}
                onMouseEnter={() => setHoveredOptionId(option.id)}
                onMouseLeave={() => setHoveredOptionId(null)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    if (!showFeedback) onOptionToggle(option.id);
                  }
                }}
                disabled={showFeedback}
                aria-pressed={selectedOptionIds.includes(option.id)}
                tabIndex={0}
                style={{
                  textAlign: 'left',
                  padding: optionPadding,
                  backgroundColor: styles.bg,
                  border: `2px solid ${styles.border}`,
                  borderRadius: '8px',
                  cursor: showFeedback ? 'default' : 'pointer',
                  color: styles.text,
                  transition: 'all 0.15s ease',
                  outline: 'none',
                  marginBottom: index < question.options.length - 1 ? (isMobile ? '8px' : '12px') : 0,
                  display: 'flex',
                  alignItems: 'center',
                }}
              >
                <div style={{
                  width: isMobile ? '20px' : '22px',
                  height: isMobile ? '20px' : '22px',
                  borderRadius: '50%',
                  border: `2px solid ${indicatorStyles.border}`,
                  backgroundColor: indicatorStyles.bg,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexShrink: 0,
                  marginRight: isMobile ? '12px' : '16px',
                }}>
                  {(state === OptionState.SELECTED || state === OptionState.CORRECT) && (
                    <svg width={isMobile ? "10" : "12"} height={isMobile ? "10" : "12"} viewBox="0 0 20 20" fill="white">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                  )}
                  {state === OptionState.INCORRECT && (
                    <svg width={isMobile ? "10" : "12"} height={isMobile ? "10" : "12"} viewBox="0 0 20 20" fill="white">
                      <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                    </svg>
                  )}
                </div>
                <span style={{ fontSize: isMobile ? '14px' : '15px', lineHeight: '1.45' }}>
                  {option.text}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      <div style={{
        backgroundColor: 'white',
        borderTop: '1px solid #e2e8f0',
        padding: actionPadding,
      }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto', display: 'flex', justifyContent: 'flex-end' }}>
          {!showFeedback && onSubmit && (
            <button
              onClick={onSubmit}
              disabled={!selectedOptionIds.length || isSubmitting}
              style={{
                padding: isMobile ? '12px 24px' : '14px 32px',
                backgroundColor: !selectedOptionIds.length || isSubmitting ? '#94a3b8' : '#0f172a',
                color: 'white',
                fontSize: isMobile ? '14px' : '15px',
                fontWeight: '500',
                borderRadius: '8px',
                cursor: !selectedOptionIds.length || isSubmitting ? 'not-allowed' : 'pointer',
                border: 'none',
                transition: 'background-color 0.15s ease',
                width: isMobile ? '100%' : 'auto',
              }}
            >
              {isSubmitting ? 'Checking...' : 'Submit Answer'}
            </button>
          )}

          {showFeedback && onNext && (
            <button
              onClick={onNext}
              style={{
                padding: isMobile ? '12px 24px' : '14px 32px',
                backgroundColor: '#0f172a',
                color: 'white',
                fontSize: isMobile ? '14px' : '15px',
                fontWeight: '500',
                borderRadius: '8px',
                cursor: 'pointer',
                border: 'none',
                transition: 'background-color 0.15s ease',
                width: isMobile ? '100%' : 'auto',
              }}
            >
              {userAnswer?.questionId === question.id && userAnswer.isCorrect
                ? 'Next Question'
                : 'Continue'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default QuestionCard;
