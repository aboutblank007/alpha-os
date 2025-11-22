"use client";

import React, { useState, useEffect } from 'react';
import { Modal } from '@/components/ui/Modal';
import { Calendar, Save, Smile, Meh, Frown } from 'lucide-react';

interface JournalNoteModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  date: Date;
  existingNote?: {
    id?: string;
    content: string;
    mood?: string;
    tags?: string[];
  };
  onSave: (note: { content: string; mood?: string; tags?: string[] }) => Promise<void>;
}

const moods = [
  { value: 'confident', label: '自信', icon: Smile, color: 'text-green-500' },
  { value: 'calm', label: '平静', icon: Meh, color: 'text-blue-500' },
  { value: 'nervous', label: '紧张', icon: Frown, color: 'text-orange-500' },
  { value: 'frustrated', label: '沮丧', icon: Frown, color: 'text-red-500' },
];

export function JournalNoteModal({
  open,
  onOpenChange,
  date,
  existingNote,
  onSave,
}: JournalNoteModalProps) {
  const [content, setContent] = useState('');
  const [mood, setMood] = useState<string | undefined>();
  const [tags, setTags] = useState<string[]>([]);
  const [newTag, setNewTag] = useState('');
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (existingNote) {
      setContent(existingNote.content || '');
      setMood(existingNote.mood);
      setTags(existingNote.tags || []);
    } else {
      setContent('');
      setMood(undefined);
      setTags([]);
    }
  }, [existingNote, open]);

  const handleSave = async () => {
    if (!content.trim()) {
      alert('请输入笔记内容');
      return;
    }

    setSaving(true);
    try {
      await onSave({ content: content.trim(), mood, tags });
      onOpenChange(false);
    } catch (error) {
      console.error('保存笔记失败:', error);
      alert('保存失败,请重试');
    } finally {
      setSaving(false);
    }
  };

  const addTag = () => {
    if (newTag.trim() && !tags.includes(newTag.trim())) {
      setTags([...tags, newTag.trim()]);
      setNewTag('');
    }
  };

  const removeTag = (tag: string) => {
    setTags(tags.filter((t) => t !== tag));
  };

  const formatDate = (d: Date) => {
    return d.toLocaleDateString('zh-CN', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      weekday: 'long',
    });
  };

  return (
    <Modal
      open={open}
      onOpenChange={onOpenChange}
      title="交易笔记"
      className="max-w-2xl"
    >
      <div className="space-y-6">
        {/* 日期显示 */}
        <div className="flex items-center gap-2 text-slate-400">
          <Calendar size={18} />
          <span className="text-sm">{formatDate(date)}</span>
        </div>

        {/* 心情选择 */}
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-3">
            今日状态
          </label>
          <div className="grid grid-cols-4 gap-3">
            {moods.map((m) => {
              const Icon = m.icon;
              return (
                <button
                  key={m.value}
                  onClick={() => setMood(m.value)}
                  className={`flex flex-col items-center gap-2 p-3 rounded-lg border-2 transition-all ${
                    mood === m.value
                      ? 'border-accent-primary bg-accent-primary/10'
                      : 'border-surface-border hover:border-slate-600'
                  }`}
                >
                  <Icon size={24} className={mood === m.value ? 'text-accent-primary' : m.color} />
                  <span className={`text-sm ${mood === m.value ? 'text-white' : 'text-slate-400'}`}>
                    {m.label}
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        {/* 笔记内容 */}
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">
            笔记内容
          </label>
          <textarea
            value={content}
            onChange={(e) => setContent(e.target.value)}
            placeholder="记录今日的交易心得、市场观察、策略调整..."
            className="w-full h-48 bg-black/20 border border-surface-border rounded-xl p-4 text-slate-300 text-sm focus:outline-none focus:border-accent-primary transition-colors resize-none"
          />
          <div className="flex justify-between text-xs text-slate-500 mt-1">
            <span>支持 Markdown 格式</span>
            <span>{content.length} 字符</span>
          </div>
        </div>

        {/* 标签 */}
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">
            标签
          </label>
          <div className="flex flex-wrap gap-2 mb-3">
            {tags.map((tag) => (
              <span
                key={tag}
                className="inline-flex items-center gap-1 px-3 py-1 bg-accent-primary/20 text-accent-primary rounded-lg text-sm"
              >
                {tag}
                <button
                  onClick={() => removeTag(tag)}
                  className="hover:text-white transition-colors"
                >
                  ×
                </button>
              </span>
            ))}
          </div>
          <div className="flex gap-2">
            <input
              type="text"
              value={newTag}
              onChange={(e) => setNewTag(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && addTag()}
              placeholder="添加标签..."
              className="flex-1 px-3 py-2 bg-black/20 border border-surface-border rounded-lg text-slate-300 text-sm focus:outline-none focus:border-accent-primary transition-colors"
            />
            <button
              onClick={addTag}
              className="px-4 py-2 bg-surface-glass border border-surface-border rounded-lg text-slate-300 hover:text-white hover:bg-white/5 transition-all text-sm"
            >
              添加
            </button>
          </div>
          <div className="flex gap-2 mt-2">
            {['趋势交易', '突破', '回调', '反转', '震荡'].map((tag) => (
              <button
                key={tag}
                onClick={() => !tags.includes(tag) && setTags([...tags, tag])}
                className="px-2 py-1 text-xs text-slate-500 hover:text-accent-primary hover:bg-accent-primary/10 rounded transition-all"
                disabled={tags.includes(tag)}
              >
                + {tag}
              </button>
            ))}
          </div>
        </div>

        {/* 操作按钮 */}
        <div className="flex justify-end gap-3 pt-4 border-t border-surface-border">
          <button
            onClick={() => onOpenChange(false)}
            className="px-4 py-2 text-slate-400 hover:text-white transition-colors"
            disabled={saving}
          >
            取消
          </button>
          <button
            onClick={handleSave}
            disabled={saving || !content.trim()}
            className="flex items-center gap-2 px-6 py-2 bg-accent-primary hover:bg-accent-primary/90 text-white rounded-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Save size={18} />
            {saving ? '保存中...' : '保存笔记'}
          </button>
        </div>
      </div>
    </Modal>
  );
}

