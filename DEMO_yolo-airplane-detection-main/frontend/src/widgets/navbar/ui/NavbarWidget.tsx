import React from 'react';
import { Menu } from 'antd';
import { useNavigate, useLocation } from 'react-router-dom';
import { HomeOutlined, HistoryOutlined, DashboardOutlined } from '@ant-design/icons';
import { ROUTES } from '../../../shared/config/routes';

const items = [
  { key: ROUTES.DETECTION, icon: <HomeOutlined />, label: 'Детекция' },
  { key: ROUTES.HISTORY, icon: <HistoryOutlined />, label: 'История' },
];

export const NavbarWidget: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const handleMenuClick = ({ key }: { key: string }) => {
    navigate(key);
  };

  return (
    <Menu
      mode="horizontal"
      selectedKeys={[location.pathname]}
      items={items}
      onClick={handleMenuClick}
      style={{ marginBottom: 0 }}
    />
  );
};
