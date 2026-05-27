import React from 'react';
import { Stage, Layer, Image as KonvaImage, Rect, Text } from 'react-konva';
import type { BoundingBox } from '../../../shared/types';

interface DetectionCanvasProps {
  image: HTMLImageElement | null;
  detections: BoundingBox[];
  selectedBox: BoundingBox | null;
  onBoxClick: (box: BoundingBox) => void;
}

const COLORS = ['#FF0000', '#00FF00', '#0000FF', '#FFA500', '#800080', '#00FFFF', '#FF00FF', '#FFFF00'];

export const DetectionCanvas: React.FC<DetectionCanvasProps> = ({
  image,
  detections,
  selectedBox,
  onBoxClick,
}) => {
  if (!image) {
    return <div style={{ textAlign: 'center', padding: '40px', color: '#999' }}>Загрузите изображение</div>;
  }

  return (
    <Stage width={image.width} height={image.height}>
      <Layer>
        <KonvaImage image={image} />
        {detections.map((box, index) => (
          <React.Fragment key={index}>
            <Rect
              x={box.x}
              y={box.y}
              width={box.width}
              height={box.height}
              stroke={COLORS[index % COLORS.length]}
              strokeWidth={selectedBox === box ? 4 : 2}
              onClick={() => onBoxClick(box)}
            />
            <Text
              x={box.x}
              y={box.y - 20}
              text={`${box.class} ${(box.confidence * 100).toFixed(1)}%`}
              fill={COLORS[index % COLORS.length]}
              fontSize={14}
              fontStyle="bold"
            />
          </React.Fragment>
        ))}
      </Layer>
    </Stage>
  );
};
