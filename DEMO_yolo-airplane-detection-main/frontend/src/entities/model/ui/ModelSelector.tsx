import React from 'react';
import { Radio } from 'antd';
import { useModelStore } from '../model';

export const ModelSelector: React.FC = () => {
  const { models, selectedModel, setSelectedModel } = useModelStore();

  if (models.length === 0) return null;

  return (
    <Radio.Group
      value={selectedModel?.id}
      onChange={(e) => {
        const model = models.find((m) => m.id === e.target.value);
        if (model) setSelectedModel(model);
      }}
    >
      {models.map((model) => (
        <Radio.Button key={model.id} value={model.id}>
          {model.name}
        </Radio.Button>
      ))}
    </Radio.Group>
  );
};
