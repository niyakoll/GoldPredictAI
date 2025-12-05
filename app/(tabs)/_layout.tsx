// app/(tabs)/_layout.tsx - Dark Bottom Tabs (Predict = Home)
import { Tabs } from 'expo-router';
import { Text } from 'react-native';

export default function TabLayout() {
  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarStyle: {
          backgroundColor: '#0f0f1e',
          borderTopColor: '#333',
          height: 84,
          paddingBottom: 28,
          paddingTop: 12,
        },
        tabBarLabelStyle: {
          fontSize: 12,
          fontWeight: '600',
        },
        tabBarActiveTintColor: '#00d4ff',
        tabBarInactiveTintColor: '#666',
      }}
    >
      {/* 1. Predict Tab - This is now the HOME page */}
      <Tabs.Screen
        name="predict"
        options={{
          title: 'Predict',
          tabBarIcon: ({ color }) => <Text style={{ color, fontSize: 24 }}>📈</Text>,
        }}
      />

      {/* 2. Community Tab */}
      <Tabs.Screen
        name="community"
        options={{
          title: 'Community',
          tabBarIcon: ({ color }) => <Text style={{ color, fontSize: 24 }}>👥</Text>,
        }}
      />

      {/* 3. Settings Tab */}
      <Tabs.Screen
        name="settings"
        options={{
          title: 'Settings',
          tabBarIcon: ({ color }) => <Text style={{ color, fontSize: 24 }}>⚙️</Text>,
        }}
      />
    </Tabs>
  );
}