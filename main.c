/**************************************************************************************
*		              ����������												  *
ʵ���������س���������ȷ���룬�������̵������ϵ�����
Ӳ�����ӣ�
					 P1 -->�������
					 P20--> EEPROMģ��SDA
					 P21--> EEPROMģ��SCL
					 P22-->�̵���ģ��J15
					 
					 ������̣�
					 0      1      2       3
					 
					 4      5      6       7
					 
					 8      9    ����    ȷ��
					 
					 A      B      C       D
ע�������																				  
***************************************************************************************/

#include "reg52.h"		       //���ļ��ж����˵�Ƭ����һЩ���⹦�ܼĴ���
#include "lcd.h"
#include "key.h"
#include "i2c.h"


#define u16 unsigned int 	  //���������ͽ�����������
#define u8 unsigned char

sbit AS=P1^0;               //�̵���  *******

sbit conn1=P2^1 ;			 //****
sbit conn2=P2^4 ;			 //****
sbit conn3=P2^3 ;
sbit conn4=P2^2;			  //****

/*****************************************************************************/

/*****************************************************************************/
u8 pw_num,Error_Num,PassWord_Length=6;
u8 PASSWORD[]={8,8,8,8,8,8,0,0,0,0};
u8 INPUT_PW_Tab[10];
u8 key_num,Step,Step5,Load_first_flag=0;
/*****************************************************************************/

/*******************************************************************************/
bit result_flag,Input_suc_flag;
bit List1=0; 
/*****************************************************************************/

/*******************************************************************************/
void Step_0();
void Step_1();
void Step_2();
void Step_3();
void Step_4();
	  
void Step_5();
	   void Step5_0();
	   void Step5_1();
	   void Step5_2();
	   void Step5_3();
	   void Step5_4();
	   void Step5_5();
void Step_6();
void Step_7();						//****
void Step_8();
void Step_9();
void Step_10();							//****
void Int0Init();					 //****

/********************************************************************************/
   

/*******************************************************************************/
void CipherComparison();
void input_password(bit m);
void Read_Password();

/*******************************************************************************
* �� �� ��       : main
* ��������		 : ������
* ��    ��       : ��
* ��    ��    	 : ��
*******************************************************************************/
void main()
{	
	u8 data1,a;
	//	ShowString(0x00," R&C face-recognition door ");  
	//	At24c02Write(0,0);                                                //������ʾ����
	LcdWriteCom(0x01);  //����
	for(data1=0;data1<PassWord_Length+2;data1++)
	{
		a=At24c02Read(data1)+0x30;
	 	LcdWriteData(a);	
		delay(1000);	
    }
	delay(1000);
	LcdInit();
	delay(1000);
	Step=0;
	Step5=0;
	Error_Num=0x00;
    Read_Password();	
						   //****
	Int0Init();  //	�����ⲿ�ж�0	****
	conn2=0;				 //****
	conn1=0;				//****
	conn4=0;
	while(1)
				{				AS=1;
								key_num=KeyDown();                   //��ȡ����ֵ
										switch(Step)
										{
												case 0:
												 {Step_0();break;}
												case 1:
												 {Step_1();break;}
												case 2:
												 {Step_2();break;}
												case 3:
												 {Step_3();break;}	
												case 4:
												 {Step_4();break;}	
												case 5:
												 {Step_5();break;}	
												case 6:
												 {Step_6();break;}
												case 7:
												 {Step_7();break;} //���»Ƽ� 
												case 8:
												 {Step_8();break;} //��������
												 case 9:
												 {Step_9();break;} //��������
												 case 10:
												  {Step_10();break;}
										}
				 }				
}

/*****************************************************************************/

/*******************************************************************************
* �� �� ��         : Int1Init()
* ��������		   : �����ⲿ�ж�1
* ��    ��         : ��
* ��    ��         : ��
*******************************************************************************/
void Int0Init()					   //****************
{
	//����INT0
	IT0=1;//�����س�����ʽ���½��أ�
	EX0=1;//��INT0���ж�������	
	EA=1;//�����ж�	
}


void Int0()	interrupt 0		//�ⲿ�ж�0���жϺ���	  //****************
{
	delay(1000);	 //��ʱ����
	if(key_num==0x0f)
	{
		AS=1;
		Step=0;
	}
}


/*******************************************************************************/
void Step_0()				   //********************
{
		LcdInit();
		ShowString(0x00,"face-recognition"); //****                          //��һ����ʾPechin Science 
	    ShowString(0x10,"    WELCOME!     ");            ///*****               // �ڶ�����ʾ WELCOME!  
		while(KeyDown()==0xff)Step=9;
		
}
/*****************************************************************************/

void Step_9()
{
	 LcdInit();
	ShowString(0x00," A:New B:facial "); //****                          //��һ����ʾPechin Science 
	  ShowString(0x10," C;Key D:close ");
	  while(KeyDown()==0xff)Step=10;
}

void Step_10()
{

	if(key_num==0x0c) {Step=7;}
	else  if(key_num==0x0d) {Step=8;}
	else if(key_num==0x0e) {Step=1;}
	else if(key_num==0x0f) {AS=1;Step=0;}	
	else Step=10;
}

/*******************************************************************************/
void Step_1()
{
		LcdWriteCom(0x01);                                           //����
	  ShowString(0x00,"Unlock");
		ShowString(0x0f,"<");	                                       //1602��һ����ʾunlock
	  ShowString(0x10,"Change Password");
		ShowString(0x1f," "); 	                                       // 1602�ڶ�����ʾChange Password                                                                            	
	  Step=2;	                                                      //
}
/*****************************************************************************/

/*******************************************************************************/
void Step_2()
{
	if(key_num!=0x0b)  
		{
					if((key_num==0x01) ||( key_num==0x09))                    //1����9������
					{
								List1=~List1;                           //Change Password
								if(List1==0)
									{                                                  
									ShowString(0x0f,"<");                           // Unlock            <
									ShowString(0x1f," ");	                          // Change Password  
								}
								else
								{
									ShowString(0x0f," ");                           // Unlock            
									ShowString(0x1f,"<");	                          // Change Password   <
								}
					}
		}
	else                                                           //ȷ�ϼ�����
	{
					if(List1==0){Step=3;}
					else        {Step=5;List1=0;}
	}
}
/*****************************************************************************/

/*******************************************************************************/
void Step_3()                                             //
{
				Step=4;
	            pw_num=0;
				LcdInit();
				ShowString(0x00,"Pass Word:  ");
}
/*****************************************************************************/

/*******************************************************************************/
void Step_4()                                                 
{
	   input_password(0);                                              //�������벢��*��ʾ
	   if(Input_suc_flag==1){Step=6;}                                  //����������ɽ�����һ��
      Input_suc_flag=0;                                              //�������������ɱ�־
}
/*****************************************************************************/

/*******************************************************************************/
void Step_5()                                                        //�޸�����
{
	switch(Step5)
	{
			case 0: {Step5_0();}  break;
			case 1: {Step5_1();}  break;
			case 2: {Step5_2();}  break;
			case 3: {Step5_3();}  break;
		    case 4: {Step5_4();}  break;
			case 5:	{Step5_5();}	break;
	}
}
/*****************************************************************************/

/*******************************************************************************/
void Step_6()
{
	u8 i=0;
	CipherComparison();                                //����ȶ�
	if(result_flag==1)                                 //������ȷ
		{
				LcdInit();
				ShowString(0x00,"    WELCOME!");
				AS=0;                                         //���̵���
				delay(200000);
				
	
				for(i=0;i<60;i++)	  //ѭ��100�Σ�Ҳ���Ǵ�Լ5S
				{
					AS=0;			 //�������
					delay(5000);	//��Լ��ʱ50ms
				}
				
				AS=1;	                                        //�ؼ̵���
		}
	else                                                //�������
		{
				LcdInit();
				ShowString(0x00,"Error 01!");
		}
		Step=0;
}

/****************************************************************************************
�������뺯��
���볤�ȣ�pw_num
����������ɱ�־��Input_suc_flag
*****************************************************************************************/

void Step5_0()                 
{
	  LcdWriteCom(0x01);  //����
	  ShowString (0x00,"Input PassWord:");     //1602��ʾ����������
	  Step5=1;
	  pw_num=0;
}
void Step5_1()                
{
		input_password(0);                  //   �������벢��*��ʾ
		if(Input_suc_flag==1)               //�����������
			{
							Step5=2;                  //
							Input_suc_flag=0;         //�������������ɱ�־
			}
}
void Step5_2()                //
{
     CipherComparison();                //����ȶ�
	   Step5=3;
}
void Step5_3()                //
{
    if(result_flag==0)         //        �������
				{
							if(Error_Num<3)             //����������С��3
							{
								Error_Num++;
								LcdInit();
								ShowString (0x00,"Error 01");
								delay(20000);
								Step5=0;
							}
							else                          //��������������3
							{
								Error_Num=0;
								Step=0;
							}					
				}
		else						      					       //������ȷ
				{
					LcdInit();
					ShowString (0x00,"New PassWord:");	
					pw_num=0;
					Step5=4;
				}  					
}
void Step5_4()
{
		input_password(1);     					       //�������벢��ʾ
		if(Input_suc_flag==1)                  //�������
			{ 
				Step5=5;
				Input_suc_flag=0;
				LcdWriteCom(0x01);  //����
	      ShowString (0x00,"      OK!");
			}
}
void Step5_5()
{
	unsigned char j;   	
	PassWord_Length=pw_num;                         //��ȡ�������볤��
	At24c02Write(0,Load_first_flag);        
  delay(100);	
	At24c02Write(1,PassWord_Length);                //���� ���볤��
	delay(100);
	for(j=0;j<PassWord_Length;j++)         
	{
		PASSWORD[j]=INPUT_PW_Tab[j];               		 //��ȡ����
		At24c02Write(j+2,INPUT_PW_Tab[j]);         		 //����������EEPROM
		delay(100);
}
	Step5=0;
	Step=0;
}


void Read_Password()
{
	unsigned char j;
	Load_first_flag=At24c02Read(0x00);
//	if(Load_first_flag==0)         		 //��������  ��ʼ���������Խ��˾�����±�������
	{
		Load_first_flag=1;
		At24c02Write(0,0x01);
		delay(100);
		At24c02Write(1,0x06);           //дĬ�����볤��6��EEPROM
		delay(100);
		for(j=0;j<PassWord_Length;j++)
		{
			At24c02Write(j+2,8);              //дĬ������888888��EEPROM
			PASSWORD[j]=INPUT_PW_Tab[j];     //������
			delay(100);
		}
	}	
	Load_first_flag=At24c02Read(0x00);
	PassWord_Length=At24c02Read(0x01);  //��ȡ���볤��
	for(j=0;j<PassWord_Length;j++)       //��ȡ����
	{
		PASSWORD[j]=At24c02Read(j+2);
	}
	
}
void input_password(bit m)
{
	unsigned char j;
	if(key_num!=0x0b)                                      //ok��û�а���
		{
					if(key_num<0x0a)                               //1-9����
					{
						INPUT_PW_Tab[pw_num]=key_num;                //������������������
						pw_num=pw_num+1;                              //���볤��+1
						 LcdWriteCom(0xc0);
						 for(j=0;j<pw_num;j++)
								{
												if(m==0) {LcdWriteData('*');	}                        //��������
												else     {LcdWriteData(INPUT_PW_Tab[j]+0x30);}         //��ʾ����
								}
											
					}
					if(key_num==0x0a)                              //���ؼ�����
					{
						if(pw_num!=0) {pw_num=pw_num-1;}
						else          {Step=0;}
						// ShowString (0x00,"Pass Word��");  
                    LcdWriteCom(0xc0);						
										for(j=0;j<pw_num;j++)
											{
														if(m==0) {LcdWriteData('*');	}                   //��������
														else     {LcdWriteData(INPUT_PW_Tab[j]+0x30);}    //��ʾ����	
										   }
											LcdWriteData(' '); 
					}
		} 
	else                          //ok������
	  {
			if(pw_num==0)	  
					{
							Step=0;
						  LcdWriteCom(0x01);
						  ShowString (0x00,"Error 02!");
							delay(10000);
					}
			else{		
							Input_suc_flag=1; 
					}
					 //AS=0;			
		}                
}
/*******************************************************
                ����ȶ�
********************************************************/
void CipherComparison()
{ 
	u8 i,j=0;
	if(PassWord_Length==pw_num)                       //���볤�ȱȶ�
	{
		for(i=0;i<PassWord_Length;i++)                  //����ȶ�
		{	
				if(PASSWORD[i]!=INPUT_PW_Tab[i])
					{
						result_flag=0;break;                     //�������
					}
				else
					{
						result_flag=1;                          //������ȷ
					}
					  INPUT_PW_Tab[i]=0XFF;                   //������뻺������
		}
	}
	else
	{result_flag=0;}	
}





void Step_7()					//*******
{
	u8 i=0;
	LcdWriteCom(0x01);  //����
	conn1=1;
	conn2=0;
	LcdInit();
	ShowString (0x00,"Recognitng...");     //1602��ʾ������ʶ��
	while(1)
	{
		key_num=KeyDown(); 
		
		
		
		if(conn3==0) {
			Step=0;
			ShowString (0x10,"complete!");     //1602��ʾ��ʶ��ɹ�
			delay(10000);
			conn1=0;
			conn2=0;	
			Step=0;
			break;
		}

		else if(key_num==0x0b)
		{
				ShowString(0x10,"    Return!");
				AS=1;
				conn4=1;
				for(i=0;i<60;i++)	  //ѭ��100�Σ�Ҳ���Ǵ�Լ5S
				{
								 //�������
					delay(5000);	//��Լ��ʱ50ms
				}
				conn1=0;
				conn2=0;
				Step=0;
				conn4=0;
				break;
		}
			
	}
}


void Step_8()						   //******
{
	u8 i=0;
	//u16 temp=2001;
	LcdWriteCom(0x01);  //����
	
	
	ShowString (0x00,"look the camera");     //1602��ʾ������ʶ��

	conn1=0;
	conn2=1;
	while(1)
	{
			
		
		key_num=KeyDown(); 
		if(conn3==0)                                 //������ȷ
		{				
				ShowString(0x10,"    WELCOME!");
				
				AS=0;                                         //���̵���
				for(i=0;i<100;i++)	  //ѭ��100�Σ�Ҳ���Ǵ�Լ5S
				{
					AS=0;			 //�������
					delay(5000);	//��Լ��ʱ50ms
				}
				AS=1;
				delay(60000);
					                                        //�ؼ̵���
			
		}
		
		else if(key_num==0x0b)
		{
				ShowString(0x10,"    Return!");
				AS=1;
				conn4=1;
				
				conn1=0;
				conn2=0;
				for(i=0;i<60;i++)	  //ѭ��100�Σ�Ҳ���Ǵ�Լ5S
				{
								 //�������
					delay(5000);	//��Լ��ʱ50ms
				}
				Step=0;
				conn4=0;
				break;
		}	
	}
	Step=0;
}
