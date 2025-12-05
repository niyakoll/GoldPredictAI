// app/(tabs)/predict/_layout.tsx
import { Stack } from 'expo-router';

export default function PredictLayout() {
  return (
    <Stack screenOptions={{ headerShown: false }}>
      {/* This is the main tab screen */}
      <Stack.Screen name="predict" />
      
      {/* All detail pages — NO tab bar! */}
      <Stack.Screen name="gold" />
      {/* Add more later: btc, oil, etc. */}
    </Stack>
  );
}