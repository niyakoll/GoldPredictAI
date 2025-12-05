// app/_layout.tsx - Root Layout (MUST include ALL your routes!)
import { Stack } from 'expo-router';

export default function RootLayout() {
  return (
    <Stack screenOptions={{ headerShown: false }}>
      {/* Public screens */}
      <Stack.Screen name="welcome" />
      <Stack.Screen name="(auth)" />

      {/* Main app — tabs */}
      <Stack.Screen name="(tabs)" />

      {/* Detail prediction screens — NO tab bar! */}
      <Stack.Screen name="predict" />   ← THIS LINE WAS MISSING!

      {/* Optional: modals */}
      <Stack.Screen name="modal" options={{ presentation: 'modal' }} />
    </Stack>
  );
}