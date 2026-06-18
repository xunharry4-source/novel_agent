type WorldviewLike = {
  worldview_id?: string | null;
  name?: string | null;
  title?: string | null;
};

const normalize = (value?: string | null): string => {
  if (typeof value !== 'string') return '';
  return value.trim();
};

export const getWorldviewDisplayName = (worldview?: WorldviewLike | null): string => {
  return (
    normalize(worldview?.name)
    || normalize(worldview?.title)
    || normalize(worldview?.worldview_id)
    || '未命名世界观'
  );
};
